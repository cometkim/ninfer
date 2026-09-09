// Host-only check of the Qwen3.6 family YaRN table builder against reference tables computed
// independently with the documented HF `_compute_yarn_parameters` formula (beta_fast 32,
// beta_slow 1, dim 64, base 1e7, original positions 262144). References carry full
// round-trip precision, so the comparison is bit-exact. Also covers the temperature and ramp
// parameterization and its rejection cases.
#include "targets/qwen3_6/impl/runtime/rope_scaling.h"

namespace rope_scaling_ns = ninfer::targets::qwen3_6;

using namespace ninfer;

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

namespace {

int check_table(const char* label, const double (&expected)[32], float factor) {
    const ops::RopeFrequencies table = rope_scaling_ns::rope_yarn_frequencies(
        1.0e7F, 64, 262144, factor);
    int failures = 0;
    for (int i = 0; i < 32; ++i) {
        const std::uint64_t got_bits =
            *reinterpret_cast<const std::uint64_t*>(&table.inv_frequency[i]);
        const std::uint64_t want_bits =
            *reinterpret_cast<const std::uint64_t*>(&expected[i]);
        if (got_bits != want_bits) {
            std::cerr << label << ": pair " << i << " inv_frequency "
                      << table.inv_frequency[i] << " != reference " << expected[i] << '\n';
            ++failures;
        }
    }
    return failures;
}

int check_attention_factor(const char* label, float factor, double expected) {
    const ops::RopeFrequencies table = rope_scaling_ns::rope_yarn_frequencies(
        1.0e7F, 64, 262144, factor);
    const double got = static_cast<double>(table.attention_factor);
    // The float storage of the attention factor rounds the exact 0.1*ln(f)+1 double.
    if (std::abs(got - expected) > 1e-7 * expected) {
        std::cerr << label << ": attention factor " << got << " != " << expected << '\n';
        return 1;
    }
    return 0;
}

// YaRN factor 4 (dim 64, base 1e7, original 262144): ramp low=14, high=22; pairs 0..14
// extrapolate, 15..21 blend, 22..31 interpolate at inv/4. Matches ROADMAP-1m-context.md §6.
constexpr double kYarn4[32] = {
    1.00000000000000000e+00, 6.04296390238132863e-01, 3.65174127254837722e-01,
    2.20673406908458991e-01, 1.33352143216332403e-01, 8.05842187761481865e-02,
    4.86967525165863113e-02, 2.94272717620928173e-02, 1.77827941003892293e-02,
    1.07460782832131743e-02, 6.49381631576211298e-03, 3.92418975848453627e-03,
    2.37137370566165538e-03, 1.43301257023696268e-03, 8.65964323360065387e-04,
    4.74239822680104593e-04, 2.56935059888680839e-04, 1.37349745076000397e-04,
    7.21738740430911334e-05, 3.70722498206803983e-05, 1.84492220250004719e-05,
    8.75977007117900255e-06, 3.84981631514872963e-06, 2.32643010232424761e-06,
    1.40585331297587280e-06, 8.49552082235639817e-07, 5.13381256614286516e-07,
    3.10234440187929882e-07, 1.87473552333113962e-07, 1.13289590940020448e-07,
    6.84604908566090348e-08, 4.13704274985795338e-08,
};

// YaRN factor 2, same ramp.
constexpr double kYarn2[32] = {
    1.00000000000000000e+00, 6.04296390238132863e-01, 3.65174127254837722e-01,
    2.20673406908458991e-01, 1.33352143216332403e-01, 8.05842187761481865e-02,
    4.86967525165863113e-02, 2.94272717620928173e-02, 1.77827941003892293e-02,
    1.07460782832131743e-02, 6.49381631576211298e-03, 3.92418975848453627e-03,
    2.37137370566165538e-03, 1.43301257023696268e-03, 8.65964323360065387e-04,
    4.90592920013901306e-04, 2.76699295264733170e-04, 1.55264929216348299e-04,
    8.66086488517093628e-05, 4.79758527091158151e-05, 2.63560314642863898e-05,
    1.43341692073838242e-05, 7.69963263029745927e-06, 4.65286020464849521e-06,
    2.81170662595174560e-06, 1.69910416447127963e-06, 1.02676251322857303e-06,
    6.20468880375859763e-07, 3.74947104666227924e-07, 2.26579181880040896e-07,
    1.36920981713218070e-07, 8.27408549971590677e-08,
};

// Linear checkpoint table, float-rounded entries of theta^(-2i/64): identical to the baked
// kernel constant table this Op shipped before frequencies became runtime values.
constexpr float kLinearText[32] = {
    1.000000000e+00F, 6.042963902e-01F, 3.651741273e-01F, 2.206734069e-01F, 1.333521432e-01F,
    8.058421878e-02F, 4.869675252e-02F, 2.942727176e-02F, 1.778279410e-02F, 1.074607828e-02F,
    6.493816316e-03F, 3.924189758e-03F, 2.371373706e-03F, 1.433012570e-03F, 8.659643234e-04F,
    5.232991147e-04F, 3.162277660e-04F, 1.910952975e-04F, 1.154781985e-04F, 6.978305849e-05F,
    4.216965034e-05F, 2.548296748e-05F, 1.539926526e-05F, 9.305720409e-06F, 5.623413252e-06F,
    3.398208329e-06F, 2.053525026e-06F, 1.240937761e-06F, 7.498942093e-07F, 4.531583638e-07F,
    2.738419634e-07F, 1.654817100e-07F,
};

int check_linear_table() {
    const ops::RopeFrequencies table = ops::rope_linear_frequencies(1.0e7F, 64);
    int failures                     = 0;
    for (int i = 0; i < 32; ++i) {
        const float got = static_cast<float>(table.inv_frequency[i]);
        if (got != kLinearText[i]) {
            std::cerr << "linear table: pair " << i << " float " << got << " != legacy "
                      << kLinearText[i] << '\n';
            ++failures;
        }
    }
    if (table.attention_factor != 1.0F) {
        std::cerr << "linear table: attention factor must stay 1\n";
        ++failures;
    }
    return failures;
}

// DFlash double ladder theta^(-i/64) at 17-digit round-trip precision. Pair 3 carries the
// correctly-rounded value; the baked pre-table constant was ~2e-13 low (a generation slip,
// invisible at BF16 rope outputs) and is superseded here.
constexpr double kLinearDflash[64] = {
    1.00000000000000000e+00, 7.77365030238775789e-01, 6.04296390238132863e-01,
    4.69758881670649164e-01, 3.65174127254837722e-01, 2.83873596475875456e-01,
    2.20673406908458991e-01, 1.71543789634287902e-01, 1.33352143216332403e-01,
    1.03663292843769794e-01, 8.05842187761481865e-02, 6.26433536656885587e-02,
    4.86967525165863113e-02, 3.78551524925863012e-02, 2.94272717620928173e-02,
    2.28757320031839559e-02, 1.77827941003892293e-02, 1.38237222735789964e-02,
    1.07460782832131743e-02, 8.35362546957826163e-03, 6.49381631576211298e-03,
    5.04806571666747105e-03, 3.92418975848453627e-03, 3.05052789026702539e-03,
    2.37137370566165538e-03, 1.84342299240911056e-03, 1.43301257023696268e-03,
    1.11397385999480246e-03, 8.65964323360065387e-04, 6.73170382414498242e-04,
    5.23299114681494734e-04, 4.06794432108304740e-04, 3.16227766016837939e-04,
    2.45824406892019762e-04, 1.91095297497044048e-04, 1.48550801717277505e-04,
    1.15478198468945822e-04, 8.97687132447314224e-05, 6.97830584859866353e-05,
    5.42469093701132573e-05, 4.21696503428582224e-05, 3.27812115139345850e-05,
    2.54829674797934641e-05, 1.98095677855033870e-05, 1.53992652605949185e-05,
    1.19708503049572999e-05, 9.30572040929699043e-06, 7.23394162736674728e-06,
    5.62341325190349121e-06, 4.37144481261108992e-06, 3.39820832894255927e-06,
    2.64164832038609264e-06, 2.05352502645714607e-06, 1.59633854428794220e-06,
    1.24093776075171953e-06, 9.64661619911199141e-07, 7.49894209332455848e-07,
    5.82941534713607427e-07, 4.53158363760081793e-07, 3.52269465147310129e-07,
    2.73841963426436139e-07, 2.12875166179637264e-07, 1.65481709994318135e-07,
    1.28639694493697462e-07,
};

int check_dflash_table() {
    const ops::RopeFrequencies table = ops::rope_linear_frequencies(1.0e7F, 128);
    int failures                     = 0;
    for (int i = 0; i < 64; ++i) {
        const double got  = table.inv_frequency[i];
        const double want = kLinearDflash[i];
        // pow varies by at most one ulp across the toolchains that generated the reference.
        const bool within_one_ulp =
            got == want || std::nextafter(want, 0.0) == got || std::nextafter(want, 1e30) == got;
        if (!within_one_ulp) {
            std::cerr << "dflash table: pair " << i << " " << got << " != legacy " << want
                      << " (beyond one ulp)\n";
            ++failures;
        }
    }
    return failures;
}

// Vision legacy float ladder theta^(-2j/36), wrapped every 18 pairs.
constexpr float kLinearVision[18] = {
    1.000000000e+00F, 5.994842503e-01F, 3.593813664e-01F, 2.154434690e-01F, 1.291549665e-01F,
    7.742636827e-02F, 4.641588834e-02F, 2.782559402e-02F, 1.668100537e-02F, 1.000000000e-02F,
    5.994842503e-03F, 3.593813664e-03F, 2.154434690e-03F, 1.291549665e-03F, 7.742636827e-04F,
    4.641588834e-04F, 2.782559402e-04F, 1.668100537e-04F,
};

int check_vision_table() {
    const ops::RopeFrequencies table = ops::rope_vision_frequencies(10'000.0F);
    int failures                     = 0;
    for (int pair = 0; pair < 36; ++pair) {
        const float got = static_cast<float>(table.inv_frequency[pair]);
        if (got != kLinearVision[pair % 18]) {
            std::cerr << "vision table: pair " << pair << " float " << got << " != legacy "
                      << kLinearVision[pair % 18] << '\n';
            ++failures;
        }
    }
    return failures;
}

// The temperature knob rewrites only the attention factor: t=0.25 at factor 2 gives
// 0.25*ln(2)+1 with the default table untouched.
int check_temperature() {
    const ops::RopeFrequencies table =
        rope_scaling_ns::rope_yarn_frequencies(1.0e7F, 64, 262144, 2.0F, 0.25F);
    const double expected = 0.25 * std::log(2.0) + 1.0;
    int failures          = 0;
    if (std::abs(static_cast<double>(table.attention_factor) - expected) > 1e-7 * expected) {
        std::cerr << "temperature: attention factor " << table.attention_factor << " != "
                  << expected << '\n';
        ++failures;
    }
    const ops::RopeFrequencies reference = rope_scaling_ns::rope_yarn_frequencies(1.0e7F, 64,
                                                                                  262144, 2.0F);
    for (int i = 0; i < 32; ++i) {
        if (table.inv_frequency[i] != reference.inv_frequency[i]) {
            std::cerr << "temperature: pair " << i << " changed the frequency table\n";
            ++failures;
        }
    }
    return failures;
}

// Ramp parameterization: beta_fast 32 -> 16 raises `low` from 14 to 15 (one fewer blending
// pair); beta_slow 1 -> 2 lowers `high` from 22 to 20 (two more interpolating pairs). Both
// move the ramp without disturbing the extrapolated prefix.
int check_ramp_direction() {
    int failures = 0;
    const ops::RopeFrequencies base =
        rope_scaling_ns::rope_yarn_frequencies(1.0e7F, 64, 262144, 4.0F);
    const ops::RopeFrequencies fast =
        rope_scaling_ns::rope_yarn_frequencies(1.0e7F, 64, 262144, 4.0F, 0.1F, 16.0F, 1.0F);
    const ops::RopeFrequencies slow =
        rope_scaling_ns::rope_yarn_frequencies(1.0e7F, 64, 262144, 4.0F, 0.1F, 32.0F, 2.0F);
    // Base ramp: low=14, high=22. correction(16) ~ 15.62 -> floor 15; correction(2) ~ 19.74 ->
    // ceil 20. The boundary pair itself is continuous (blend at e=0 equals extrapolation), so
    // pair 15 is the discriminator: it blends at e=1/8 under the base table but extrapolates
    // under bf=16.
    if (fast.inv_frequency[15] == base.inv_frequency[15]) {
        std::cerr << "ramp: bf=16 left the ramp unchanged\n";
        ++failures;
    }
    if (fast.inv_frequency[13] != base.inv_frequency[13]) {
        std::cerr << "ramp: bf=16 disturbed the extrapolated prefix\n";
        ++failures;
    }
    if (slow.inv_frequency[21] == base.inv_frequency[21]) {
        std::cerr << "ramp: bs=2 left the ramp unchanged\n";
        ++failures;
    }
    if (slow.inv_frequency[14] != base.inv_frequency[14]) {
        std::cerr << "ramp: bs=2 disturbed the extrapolated prefix\n";
        ++failures;
    }
    return failures;
}

int check_rejections() {
    int failures = 0;
    const auto expect_throw = [&](const char* label, auto build) {
        try {
            build();
            std::cerr << label << ": accepted\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
    };
    expect_throw("yarn factor 1", [] {
        (void)rope_scaling_ns::rope_yarn_frequencies(1.0e7F, 64, 262144, 1.0F);
    });
    expect_throw("yarn temperature 0", [] {
        (void)rope_scaling_ns::rope_yarn_frequencies(1.0e7F, 64, 262144, 2.0F, 0.0F);
    });
    expect_throw("yarn beta_fast <= beta_slow", [] {
        (void)rope_scaling_ns::rope_yarn_frequencies(1.0e7F, 64, 262144, 2.0F, 0.1F, 1.0F, 1.0F);
    });
    expect_throw("yarn beta_slow 0", [] {
        (void)rope_scaling_ns::rope_yarn_frequencies(1.0e7F, 64, 262144, 2.0F, 0.1F, 32.0F, 0.0F);
    });
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += check_table("yarn4", kYarn4, 4.0F);
    failures += check_table("yarn2", kYarn2, 2.0F);
    failures += check_attention_factor("yarn4 attention", 4.0F, 1.13862943611198908);
    failures += check_attention_factor("yarn2 attention", 2.0F, 1.06931471805599454);
    failures += check_temperature();
    failures += check_ramp_direction();
    failures += check_linear_table();
    failures += check_dflash_table();
    failures += check_vision_table();
    failures += check_rejections();
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " rope scaling tables\n";
    return failures == 0 ? 0 : 1;
}
