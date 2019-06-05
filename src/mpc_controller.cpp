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

#include "rpg_mpc/mpc_controller.h"

#include <ctime>
#include <boost/format.hpp>
#include <fstream>

namespace rpg_mpc
{

template <typename T>
MpcController<T>::MpcController(MpcParams<T> &params)
    : params_(params),
      mpc_wrapper_(MpcWrapper<T>()),
      timing_feedback_(T(1e-3)),
      timing_preparation_(T(1e-3)),
      est_state_((Eigen::Matrix<T, kStateSize, 1>() << 0, 0, 0, 1, 0, 0, 0, 0, 0, 0).finished()),
      reference_states_(Eigen::Matrix<T, kStateSize, kSamples + 1>::Zero()),
      reference_inputs_(Eigen::Matrix<T, kInputSize, kSamples + 1>::Zero()),
      predicted_states_(Eigen::Matrix<T, kStateSize, kSamples + 1>::Zero()),
      predicted_inputs_(Eigen::Matrix<T, kInputSize, kSamples>::Zero()),
      point_of_interest_(Eigen::Matrix<T, 3, 1>::Zero())
{
    setNewParams(params_);

    preparation_thread_ = std::thread(&MpcWrapper<T>::prepare, mpc_wrapper_);
}

template <typename T>
MpcController<T>::~MpcController()
{
    if (preparation_thread_.joinable())
    {
        preparation_thread_.join();
    }
}

template <typename T>
ControlInput MpcController<T>::run(
    const CopterState &state_estimate,
    const std::list<Checkpoint> &reference_trajectory,
    const MpcParams<T> &params)
{
    double call_time = Clock::nowSeconds();
    const clock_t start = clock();

    if (params.changed_)
    {
        params_ = params;
        setNewParams(params_);
    }

    preparation_thread_.join();

    // Convert everything into Eigen format.
    setStateEstimate(state_estimate);
    setReference(reference_trajectory);

    static const bool do_preparation_step(false);

    // Get the feedback from MPC.
    mpc_wrapper_.setTrajectory(reference_states_, reference_inputs_);
    mpc_wrapper_.update(est_state_, do_preparation_step);
    mpc_wrapper_.getStates(predicted_states_);
    mpc_wrapper_.getInputs(predicted_inputs_);

    // Publish the predicted trajectory.
    publishPrediction(predicted_states_, predicted_inputs_, call_time);

    // Start a thread to prepare for the next execution.
    preparation_thread_ = std::thread(&MpcController<T>::preparationThread, this);

    // Timing
    const clock_t end = clock();
    timing_feedback_ = 0.9 * timing_feedback_ + 0.1 * double(end - start) / CLOCKS_PER_SEC;
    if (params_.print_info_)
    {
        std::cout << "MPC Timing: Latency: " << timing_feedback_ * 1000 << " ms  |  Total: " << (timing_feedback_ + timing_preparation_) * 1000 << " ms";
    }

    // Return the input control command.
    return updateControlCommand(predicted_states_.col(0),
                                predicted_inputs_.col(0),
                                call_time);
}

template <typename T>
bool MpcController<T>::setStateEstimate(
    const CopterState &state_estimate)
{
    est_state_(kPosX) = state_estimate.position.x();
    est_state_(kPosY) = state_estimate.position.y();
    est_state_(kPosZ) = state_estimate.position.z();
    est_state_(kOriW) = state_estimate.attitude.w();
    est_state_(kOriX) = state_estimate.attitude.x();
    est_state_(kOriY) = state_estimate.attitude.y();
    est_state_(kOriZ) = state_estimate.attitude.z();
    est_state_(kVelX) = state_estimate.velocity.x();
    est_state_(kVelY) = state_estimate.velocity.y();
    est_state_(kVelZ) = state_estimate.velocity.z();
    const bool quaternion_norm_ok = abs(est_state_.segment(kOriW, 4).norm() - 1.0) < 0.1;
    return quaternion_norm_ok;
}

template <typename T>
bool MpcController<T>::setReference(
    const std::list<Checkpoint> &reference_trajectory)
{
    reference_states_.setZero();
    reference_inputs_.setZero();

    const T dt = mpc_wrapper_.getTimestep();
    Eigen::Matrix<T, 3, 1> acceleration;
    const Eigen::Matrix<T, 3, 1> gravity(0.0, 0.0, -9.81);
    Eigen::Quaternion<T> q_heading;
    Eigen::Quaternion<T> q_orientation;
    bool quaternion_norm_ok(true);
    if (reference_trajectory.size() == 1)
    {
        q_heading = Eigen::Quaternion<T>(Eigen::AngleAxis<T>(reference_trajectory.front().heading, Eigen::Matrix<T, 3, 1>::UnitZ()));
        q_orientation = q_heading * reference_trajectory.front().attitude.template cast<T>();
        reference_states_ = (Eigen::Matrix<T, kStateSize, 1>() << reference_trajectory.front().position.template cast<T>(),
                             q_orientation.w(),
                             q_orientation.x(),
                             q_orientation.y(),
                             q_orientation.z(),
                             reference_trajectory.front().velocity.template cast<T>())
                                .finished()
                                .replicate(1, kSamples + 1);

        acceleration << reference_trajectory.front().acceleration.template cast<T>() - gravity;
        reference_inputs_ = (Eigen::Matrix<T, kInputSize, 1>() << acceleration.norm(),
                             reference_trajectory.front().bodyrates.template cast<T>())
                                .finished()
                                .replicate(1, kSamples + 1);
    }
    else
    {
        std::list<Checkpoint>::const_iterator iterator(reference_trajectory.begin());
        for (int i = 0; i < kSamples + 1; i++)
        {
            while (iterator->seconds < i * dt &&
                   iterator != reference_trajectory.end())
            {
                iterator++;
            }
            q_heading = Eigen::Quaternion<T>(Eigen::AngleAxis<T>(iterator->heading, Eigen::Matrix<T, 3, 1>::UnitZ()));
            q_orientation = q_heading * iterator->attitude.template cast<T>();
            reference_states_.col(i) << iterator->position.template cast<T>(),
                q_orientation.w(),
                q_orientation.x(),
                q_orientation.y(),
                q_orientation.z(),
                iterator->velocity.template cast<T>();
            if (reference_states_.col(i).segment(kOriW, 4).dot(est_state_.segment(kOriW, 4)) < 0.0)
            {
                reference_states_.block(kOriW, i, 4, 1) = -reference_states_.block(kOriW, i, 4, 1);
            }
            acceleration << iterator->acceleration.template cast<T>() - gravity;
            reference_inputs_.col(i) << acceleration.norm(), iterator->bodyrates.template cast<T>();
            quaternion_norm_ok &= abs(est_state_.segment(kOriW, 4).norm() - 1.0) < 0.1;
        }
    }
    return quaternion_norm_ok;
}

template <typename T>
ControlInput MpcController<T>::updateControlCommand(
    const Eigen::Ref<const Eigen::Matrix<T, kStateSize, 1>> state,
    const Eigen::Ref<const Eigen::Matrix<T, kInputSize, 1>> input,
    double &time)
{
    Eigen::Matrix<T, kInputSize, 1> input_bounded = input.template cast<T>();

    // Bound inputs for sanity.
    input_bounded(INPUT::kThrust) = std::max(params_.min_thrust_, std::min(params_.max_thrust_, input_bounded(INPUT::kThrust)));
    input_bounded(INPUT::kRateX) = std::max(-params_.max_bodyrate_xy_, std::min(params_.max_bodyrate_xy_, input_bounded(INPUT::kRateX)));
    input_bounded(INPUT::kRateY) = std::max(-params_.max_bodyrate_xy_, std::min(params_.max_bodyrate_xy_, input_bounded(INPUT::kRateY)));
    input_bounded(INPUT::kRateZ) = std::max(-params_.max_bodyrate_z_, std::min(params_.max_bodyrate_z_, input_bounded(INPUT::kRateZ)));

    ControlInput command;
    command.armed = true;
    command.type = ControlInput::ControlInputType::TRPY;
    command.thrust = input_bounded(INPUT::kThrust);
    command.roll = input_bounded(INPUT::kRateX);
    command.pitch = input_bounded(INPUT::kRateY);
    command.yaw = input_bounded(INPUT::kRateZ);
    return command;
}

static int count = 0;

template <typename T>
bool MpcController<T>::publishPrediction(
    const Eigen::Ref<const Eigen::Matrix<T, kStateSize, kSamples + 1>> states,
    const Eigen::Ref<const Eigen::Matrix<T, kInputSize, kSamples>> inputs,
    double &time)
{
    boost::format fmt("traj%04d");
    fmt % count++;
    std::string fname = "./export/mpc/prediction/" + fmt.str();
    std::ofstream tfile(fname);

    for (int i = 0; i <= kSamples; i++)
    {
        Eigen::Matrix<T, kStateSize, 1> state = states.template block<kStateSize, 1>(0, i);
        Eigen::Matrix<T, kInputSize, 1> input;
        if (i < kSamples)
        {
            input = inputs.template block<kInputSize, 1>(0, i);
        }
        else
        {
            input.setZero();
        }

        tfile << i << " "                                                                   // sample
              << state(0) << " " << state(1) << " " << state(2) << " "                      // pos
              << state(3) << " " << state(4) << " " << state(5) << " " << state(6) << " "   // quat
              << state(7) << " " << state(8) << " " << state(9) << " "                      // vel
              << input(0) << " " << input(1) << " " << input(2) << " " << input(3) << "\n"; // input
    }

    tfile.close();

    return true;
}

template <typename T>
void MpcController<T>::preparationThread()
{
    const clock_t start = clock();

    mpc_wrapper_.prepare();

    // Timing
    const clock_t end = clock();
    timing_preparation_ = 0.9 * timing_preparation_ +
                          0.1 * double(end - start) / CLOCKS_PER_SEC;
}

template <typename T>
bool MpcController<T>::setNewParams(MpcParams<T> &params)
{
    mpc_wrapper_.setCosts(params.Q_, params.R_);
    mpc_wrapper_.setLimits(
        params.min_thrust_, params.max_thrust_,
        params.max_bodyrate_xy_, params.max_bodyrate_z_);
    params.changed_ = false;
    return true;
}

template class MpcController<float>;
template class MpcController<double>;

} // namespace rpg_mpc
