/*    rpg_quadrotor_mpc
 *    A model predictive control implementation for quadrotors.
 *    Copyright (C) 2017-2018 Philipp Foehn, 
 *    Robotics and Perception Group, University of Zurich
 * 
 *    Intended to be used with rpg_quadrotor_control and rpg_quadrotor_common.
 *    https://github.com/uzh-rpg/rpg_quadrotor_control
 *
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#pragma once

#include <thread>

#include <Eigen/Eigen>
#include <core/controlinput.h>
#include <core/checkpoint.h>
#include <core/copterstate.h>
#include <timing/clock.h>

#include "rpg_mpc/mpc_wrapper.h"
#include "rpg_mpc/mpc_params.h"

namespace rpg_mpc
{

enum STATE
{
    kPosX = 0,
    kPosY = 1,
    kPosZ = 2,
    kOriW = 3,
    kOriX = 4,
    kOriY = 5,
    kOriZ = 6,
    kVelX = 7,
    kVelY = 8,
    kVelZ = 9
};

enum INPUT
{
    kThrust = 0,
    kRateX = 1,
    kRateY = 2,
    kRateZ = 3
};

template <typename T>
class MpcController
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static_assert(kStateSize == 10,
                  "MpcController: Wrong model size. Number of states does not match.");
    static_assert(kInputSize == 4,
                  "MpcController: Wrong model size. Number of inputs does not match.");

    MpcController(MpcParams<T> &params);

    ~MpcController();

    ControlInput run(
        const CopterState &state_estimate,
        const std::list<Checkpoint> &reference_trajectory,
        const MpcParams<T>& params);

private:
    // Internal helper functions.

    bool setStateEstimate(
        const CopterState &state_estimate);

    bool setReference(const std::list<Checkpoint> &reference_trajectory);

    ControlInput updateControlCommand(
        const Eigen::Ref<const Eigen::Matrix<T, kStateSize, 1>> state,
        const Eigen::Ref<const Eigen::Matrix<T, kInputSize, 1>> input,
        double &time);

    bool publishPrediction(
        const Eigen::Ref<const Eigen::Matrix<T, kStateSize, kSamples + 1>> states,
        const Eigen::Ref<const Eigen::Matrix<T, kInputSize, kSamples>> inputs,
        double &time);

    void preparationThread();

    bool setNewParams(MpcParams<T> &params);

    // Parameters
    MpcParams<T> params_;

    // MPC
    MpcWrapper<T> mpc_wrapper_;

    // Preparation Thread
    std::thread preparation_thread_;

    // Variables
    T timing_feedback_, timing_preparation_;
    Eigen::Matrix<T, kStateSize, 1> est_state_;
    Eigen::Matrix<T, kStateSize, kSamples + 1> reference_states_;
    Eigen::Matrix<T, kInputSize, kSamples + 1> reference_inputs_;
    Eigen::Matrix<T, kStateSize, kSamples + 1> predicted_states_;
    Eigen::Matrix<T, kInputSize, kSamples> predicted_inputs_;
    Eigen::Matrix<T, 3, 1> point_of_interest_;
};

} // namespace rpg_mpc
