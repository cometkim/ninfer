"""Keep large offline transfers from retaining whole artifacts in the OS page cache."""

from __future__ import annotations

import os

# os.pwrite/posix_fadvise/fdatasync are POSIX; positional writes and flushes have portable
# stand-ins on Windows, and page-cache eviction has none (the offline writer is throughput
# planning, not correctness).
if not hasattr(os, "pwrite"):
    def _pwrite(fd: int, data, offset: int) -> int:
        import ctypes
        import msvcrt

        handle = msvcrt.get_osfhandle(fd)
        over = ctypes.create_string_buffer(0)  # placeholder, replaced below

        class _OVERLAPPED(ctypes.Structure):
            _fields_ = [("Internal", ctypes.c_ulonglong),
                        ("InternalHigh", ctypes.c_ulonglong),
                        ("Offset", ctypes.c_ulong),
                        ("OffsetHigh", ctypes.c_ulong),
                        ("hEvent", ctypes.c_void_p)]

        over = _OVERLAPPED()
        over.Offset = offset & 0xFFFFFFFF
        over.OffsetHigh = (offset >> 32) & 0xFFFFFFFF
        chunk = ctypes.create_string_buffer(bytes(data[:IO_CHUNK_BYTES]))
        written = ctypes.c_ulong(0)
        ok = ctypes.windll.kernel32.WriteFile(
            ctypes.c_void_p(handle), chunk, min(len(data), IO_CHUNK_BYTES),
            ctypes.byref(written), ctypes.byref(over))
        if not ok:
            raise ctypes.WinError()
        return written.value

    os.pwrite = _pwrite

if not hasattr(os, "pread"):
    def _pread(fd: int, count: int, offset: int) -> bytes:
        """Read binary bytes at an offset without CRT text-mode translation."""
        if count < 0 or offset < 0:
            raise ValueError("pread count and offset must be nonnegative")
        if not count:
            return b""
        # A synchronous Win32 file handle changes its current position even with
        # an OVERLAPPED offset. Restore it: our readers are single-owner and use
        # only positional reads, but callers still expect pread's offset contract.
        position = os.lseek(fd, 0, os.SEEK_CUR)
        try:
            return _read_at(fd, count, offset)
        finally:
            os.lseek(fd, position, os.SEEK_SET)

    def _read_at(fd: int, count: int, offset: int) -> bytes:
        import ctypes
        import msvcrt

        class _OVERLAPPED(ctypes.Structure):
            _fields_ = [("Internal", ctypes.c_size_t),
                        ("InternalHigh", ctypes.c_size_t),
                        ("Offset", ctypes.c_ulong),
                        ("OffsetHigh", ctypes.c_ulong),
                        ("hEvent", ctypes.c_void_p)]

        chunks = []
        while count:
            size = min(count, IO_CHUNK_BYTES)
            buffer = ctypes.create_string_buffer(size)
            over = _OVERLAPPED()
            over.Offset = offset & 0xFFFFFFFF
            over.OffsetHigh = offset >> 32
            received = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.ReadFile(
                ctypes.c_void_p(msvcrt.get_osfhandle(fd)), buffer, size,
                ctypes.byref(received), ctypes.byref(over))
            if not ok:
                error = ctypes.windll.kernel32.GetLastError()
                if error == 38:  # ERROR_HANDLE_EOF
                    break
                raise ctypes.WinError(error)
            chunks.append(buffer.raw[:received.value])
            if received.value < size:
                break
            count -= received.value
            offset += received.value
        return b"".join(chunks)

    os.pread = _pread

if not hasattr(os, "posix_fadvise"):
    def _fadvise_noop(fd: int, offset: int, count: int, advise: int) -> None:
        return

    os.posix_fadvise = _fadvise_noop
    os.POSIX_FADV_DONTNEED = 4

if not hasattr(os, "fdatasync"):
    os.fdatasync = os.fsync

IO_CHUNK_BYTES = 8 * 1024 * 1024
WRITEBACK_BYTES = 64 * 1024 * 1024
if hasattr(os, "sysconf"):
    _PAGE_BYTES = os.sysconf("SC_PAGE_SIZE")
else:  # Windows has no sysconf; the 4096-byte direct-I/O unit is fixed by the container contract
    _PAGE_BYTES = 4096


def discard_cached_pages(fd: int, offset: int = 0, count: int | None = None) -> None:
    if count is None:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    elif count > 0:
        begin = offset // _PAGE_BYTES * _PAGE_BYTES
        end = (offset + count + _PAGE_BYTES - 1) // _PAGE_BYTES * _PAGE_BYTES
        os.posix_fadvise(fd, begin, end - begin, os.POSIX_FADV_DONTNEED)


class Writeback:
    """Bound dirty output across all open shards; release clean pages after writeback."""

    def __init__(self) -> None:
        self._bytes = 0
        self._fds: set[int] = set()

    def written(self, fd: int, count: int) -> None:
        self._fds.add(fd)
        self._bytes += count
        if self._bytes >= WRITEBACK_BYTES:
            self.flush()

    def flush(self) -> None:
        for fd in self._fds:
            os.fdatasync(fd)
            discard_cached_pages(fd)
        self._fds.clear()
        self._bytes = 0
