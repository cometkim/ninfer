#include "artifact/file_io.h"

#include "artifact/framing.h"
#include "artifact/schema.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ninfer::artifact {
namespace {

#ifdef _WIN32

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation, DWORD error) {
    throw ArtifactError(path.string() + ": " + operation + ": error " + std::to_string(error));
}

#else

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw ArtifactError(path.string() + ": " + operation + ": " + std::strerror(errno));
}

off_t file_offset(std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw ArtifactError("file offset exceeds positional I/O range");
    }
    return static_cast<off_t>(offset);
}

#endif

} // namespace

#ifdef _WIN32

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
    fd_ = reinterpret_cast<intptr_t>(::CreateFileW(path_.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                                   nullptr, OPEN_EXISTING,
                                                   FILE_ATTRIBUTE_NORMAL, nullptr));
    if (fd_ == kInvalidHandle) { fail(path_, "open", ::GetLastError()); }

    LARGE_INTEGER size{};
    if (!::GetFileSizeEx(reinterpret_cast<HANDLE>(fd_), &size)) {
        const auto error = ::GetLastError();
        ::CloseHandle(reinterpret_cast<HANDLE>(fd_));
        fd_   = kInvalidHandle;
        fail(path_, "GetFileSizeEx", error);
    }
    if (size.QuadPart < 0) {
        ::CloseHandle(reinterpret_cast<HANDLE>(fd_));
        fd_ = kInvalidHandle;
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = static_cast<std::uint64_t>(size.QuadPart);
}

InputFile::~InputFile() {
    if (direct_fd_ != kInvalidHandle) {
        ::CloseHandle(reinterpret_cast<HANDLE>(direct_fd_));
    }
    if (fd_ != kInvalidHandle) { ::CloseHandle(reinterpret_cast<HANDLE>(fd_)); }
}

namespace {

// Positional read through the shared synchronous handle. ReadFile on a synchronous
// handle opened without FILE_FLAG_OVERLAPPED still requires an OVERLAPPED structure
// to carry the 64-bit offset.
std::size_t positional_read(HANDLE file, std::uint64_t offset, std::span<std::byte> destination,
                            const std::filesystem::path& path) {
    OVERLAPPED operation{};
    operation.Offset     = static_cast<DWORD>(offset & 0xffffffffULL);
    operation.OffsetHigh = static_cast<DWORD>(offset >> 32U);
    DWORD read           = 0;
    if (!::ReadFile(file, destination.data(), static_cast<DWORD>(destination.size()), &read,
                    &operation)) {
        fail(path, "ReadFile", ::GetLastError());
    }
    return read;
}

} // namespace

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        const auto read = positional_read(reinterpret_cast<HANDLE>(fd_), offset,
                                          destination.first(count), path_);
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += static_cast<std::uint64_t>(read);
        destination = destination.subspan(static_cast<std::size_t>(read));
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<DWORD>::max())) {
        throw ArtifactError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
    if (direct_fd_ == kInvalidHandle) {
        // The Windows counterpart of POSIX O_DIRECT: unbuffered overlapped I/O with
        // sector-aligned offsets and buffers, satisfying the same 4096-byte contract.
        direct_fd_ = reinterpret_cast<intptr_t>(
            ::CreateFileW(path_.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                          FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING |
                              FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN,
                          nullptr));
        if (direct_fd_ == kInvalidHandle) { fail(path_, "open direct", ::GetLastError()); }
    }
    return positional_read(reinterpret_cast<HANDLE>(direct_fd_), offset, destination, path_);
}

#else

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
    fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) { fail(path_, "open"); }

    struct stat status {};

    if (::fstat(fd_, &status) != 0) {
        const auto error = errno;
        ::close(fd_);
        fd_   = -1;
        errno = error;
        fail(path_, "fstat");
    }
    if (status.st_size < 0 || !S_ISREG(status.st_mode)) {
        ::close(fd_);
        fd_ = -1;
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = static_cast<std::uint64_t>(status.st_size);
}

InputFile::~InputFile() {
    if (direct_fd_ >= 0) { ::close(direct_fd_); }
    if (fd_ >= 0) { ::close(fd_); }
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        const auto read  = ::pread(fd_, destination.data(), count, file_offset(offset));
        if (read < 0) {
            if (errno == EINTR) { continue; }
            fail(path_, "pread");
        }
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += static_cast<std::uint64_t>(read);
        destination = destination.subspan(static_cast<std::size_t>(read));
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
        throw ArtifactError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
    if (direct_fd_ < 0) {
        direct_fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (direct_fd_ < 0) { fail(path_, "open direct"); }
    }
    ssize_t read;
    do {
        read = ::pread(direct_fd_, destination.data(), destination.size(), file_offset(offset));
    } while (read < 0 && errno == EINTR);
    if (read < 0) { fail(path_, "direct pread"); }
    return static_cast<std::size_t>(read);
}

#endif

} // namespace ninfer::artifact
