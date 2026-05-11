#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "../../include/cost.h"

namespace {

std::vector<float> probabilities_for_zero_logits(int num_classes,
                                                 bool use_prior_bias = true) {
    HRCSoftmax hs = class_to_obs(num_classes, use_prior_bias);
    std::vector<float> mz(hs.len, 0.0f);
    std::vector<float> Sz(hs.len, 0.0f);
    return obs_to_class(mz, Sz, hs, num_classes);
}

float probability_sum_for_zero_logits(int num_classes) {
    auto probs = probabilities_for_zero_logits(num_classes);
    return std::accumulate(probs.begin(), probs.end(), 0.0f);
}

}  // namespace

TEST(HRCSoftmax, ZeroLogitsNormalizeForAnyClassCount) {
    for (int num_classes : {2, 3, 4, 5, 8, 10, 16, 100, 1000}) {
        EXPECT_NEAR(probability_sum_for_zero_logits(num_classes), 1.0f, 1e-6f)
            << "num_classes=" << num_classes;
    }
}

TEST(HRCSoftmax, PriorBiasMakesZeroLogitsUniform) {
    for (int num_classes : {2, 3, 4, 5, 8, 10, 16, 100, 1000}) {
        auto probs = probabilities_for_zero_logits(num_classes);
        const float expected = 1.0f / static_cast<float>(num_classes);
        for (float prob : probs) {
            EXPECT_NEAR(prob, expected, 1e-6f)
                << "num_classes=" << num_classes;
        }
    }
}

TEST(HRCSoftmax, PriorBiasCanBeDisabled) {
    HRCSoftmax hs = class_to_obs(10, false);
    EXPECT_TRUE(hs.bias.empty());

    auto probs = probabilities_for_zero_logits(10, false);
    auto minmax = std::minmax_element(probs.begin(), probs.end());
    EXPECT_GT(*minmax.second - *minmax.first, 1e-3f);
}

TEST(HRCSoftmax, BuildsExactLeafTree) {
    HRCSoftmax hs = class_to_obs(10);

    EXPECT_EQ(hs.len, 9);
    EXPECT_EQ(hs.path_len.size(), 10);
    EXPECT_EQ(hs.obs.size(), hs.idx.size());
    EXPECT_EQ(hs.obs.size(), 10 * hs.n_obs);
    EXPECT_TRUE(std::all_of(hs.path_len.begin(), hs.path_len.end(),
                            [](int len) { return len > 0; }));
}
