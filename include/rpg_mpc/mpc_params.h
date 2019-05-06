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

#include <rpg_mpc/mpc_wrapper.h>
#include "config/mpcconfig.h"

namespace rpg_mpc
{

template <typename T>
class MpcParams
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MpcParams() : changed_(false),
                  print_info_(false),
                  state_cost_exponential_(0.0),
                  input_cost_exponential_(0.0),
                  max_bodyrate_xy_(0.0),
                  max_bodyrate_z_(0.0),
                  min_thrust_(0.0),
                  max_thrust_(0.0),
                  p_B_C_(Eigen::Matrix<T, 3, 1>::Zero()),
                  q_B_C_(Eigen::Quaternion<T>(1.0, 0.0, 0.0, 0.0)),
                  Q_(Eigen::Matrix<T, kCostSize, kCostSize>::Zero()),
                  R_(Eigen::Matrix<T, kInputSize, kInputSize>::Zero())
    {
    }

    ~MpcParams()
    {
    }

    bool loadParameters(MpcConfig &config)
    {
        // Read state costs.
        T Q_pos_xy = (T)config.Q_pos_xy;
        T Q_pos_z = (T)config.Q_pos_z;
        T Q_attitude = (T)config.Q_attitude;
        T Q_velocity = (T)config.Q_velocity;
        T Q_perception = (T)config.Q_perception;

        // Check whether all state costs are positive.
        if (Q_pos_xy <= 0.0 ||
            Q_pos_z <= 0.0 ||
            Q_attitude <= 0.0 ||
            Q_velocity <= 0.0 ||
            Q_perception < 0.0) // Perception cost can be zero to deactivate.
        {
            std::cerr << "MPC: State cost Q has negative enries!\n";
            return false;
        }

        // Read input costs.
        T R_thrust = (T)config.R_thrust;
        T R_pitchroll = (T)config.R_pitchroll;
        T R_yaw = (T)config.R_yaw;

        // Check whether all input costs are positive.
        if (R_thrust <= 0.0 ||
            R_pitchroll <= 0.0 ||
            R_yaw <= 0.0)
        {
            std::cerr << "MPC: Input cost R has negative enries!\n";
            return false;
        }

        // Set state and input cost matrices.
        Q_ = (Eigen::Matrix<T, kCostSize, 1>() << Q_pos_xy, Q_pos_xy, Q_pos_z,
              Q_attitude, Q_attitude, Q_attitude, Q_attitude,
              Q_velocity, Q_velocity, Q_velocity,
              Q_perception, Q_perception)
                 .finished()
                 .asDiagonal();
        R_ = (Eigen::Matrix<T, kInputSize, 1>() << R_thrust, R_pitchroll, R_pitchroll, R_yaw).finished().asDiagonal();

        // Read cost scaling values
        state_cost_exponential_ = (T)config.state_cost_exponential;
        input_cost_exponential_ = (T)config.input_cost_exponential;

        // Read input limits.
        max_bodyrate_xy_ = (T)config.max_bodyrate_xy;
        max_bodyrate_z_ = (T)config.max_bodyrate_z;
        min_thrust_ = (T)config.min_thrust;
        max_thrust_ = (T)config.max_thrust;

        // Check whether all input limits are positive.
        if (max_bodyrate_xy_ <= 0.0 ||
            max_bodyrate_z_ <= 0.0 ||
            min_thrust_ <= 0.0 ||
            max_thrust_ <= 0.0)
        {
            std::cerr << "MPC: All limits must be positive non-zero values!\n";
            return false;
        }

        // Optional parameters
        std::vector<T> p_B_C(3), q_B_C(4);
        p_B_C_ = config.p_B_C.cast<T>();
        q_B_C_ = config.q_B_C.cast<T>();

        print_info_ = config.print_info;
        if (print_info_)
            std::cout << "MPC: Informative printing enabled.\n";

        changed_ = true;

        return true;
    }

    bool changed_;

    bool print_info_;

    T state_cost_exponential_;
    T input_cost_exponential_;

    T max_bodyrate_xy_;
    T max_bodyrate_z_;
    T min_thrust_;
    T max_thrust_;

    Eigen::Matrix<T, 3, 1> p_B_C_;
    Eigen::Quaternion<T> q_B_C_;

    Eigen::Matrix<T, kCostSize, kCostSize> Q_;
    Eigen::Matrix<T, kInputSize, kInputSize> R_;
};

} // namespace rpg_mpc