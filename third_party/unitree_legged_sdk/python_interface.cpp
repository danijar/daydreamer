/************************************************************************
Copyright (c) 2020, Unitree Robotics.Co.Ltd. All rights reserved.
Use of this source code is governed by the MPL-2.0 license, see LICENSE.
************************************************************************/

#include <math.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>

#include "unitree_legged_sdk/unitree_legged_sdk.h"

using namespace UNITREE_LEGGED_SDK;

class RobotInterface {
 public:
  RobotInterface(uint8_t level) : safe(LeggedType::A1), udp(level) {
    // InitEnvironment();
  }
  LowState ReceiveObservation();
  HighState ReceiveHighObservation();
  void SendCommand(std::array<float, 60> motorcmd);
  void SendHighCommand(float forwardSpeed, float sideSpeed, float rotateSpeed,
                       float bodyHeight, int mode);
  void Initialize();

  UDP udp;
  Safety safe;
  LowState state = {0};
  LowCmd low_cmd = {0};  
  HighState high_state = {0};
  HighCmd high_cmd = {0};
};

LowState RobotInterface::ReceiveObservation() {
  udp.Recv();
  udp.GetRecv(state);
  return state;
}

void RobotInterface::SendCommand(std::array<float, 60> motorcmd) {
  low_cmd.levelFlag = 0xff;
  for (int motor_id = 0; motor_id < 12; motor_id++) {
    low_cmd.motorCmd[motor_id].mode = 0x0A;
    low_cmd.motorCmd[motor_id].q = motorcmd[motor_id * 5];
    low_cmd.motorCmd[motor_id].Kp = motorcmd[motor_id * 5 + 1];
    low_cmd.motorCmd[motor_id].dq = motorcmd[motor_id * 5 + 2];
    low_cmd.motorCmd[motor_id].Kd = motorcmd[motor_id * 5 + 3];
    low_cmd.motorCmd[motor_id].tau = motorcmd[motor_id * 5 + 4];
  }
  safe.PositionLimit(low_cmd);
  udp.SetSend(low_cmd);
  udp.Send();
}

HighState RobotInterface::ReceiveHighObservation() {
  udp.Recv();
  udp.GetRecv(high_state);
  return high_state;
}


void RobotInterface::SendHighCommand(float forwardSpeed, float sideSpeed, float rotateSpeed,
                       float bodyHeight, int mode) {
    high_cmd.levelFlag = 0x00;
    high_cmd.forwardSpeed = forwardSpeed;
    high_cmd.sideSpeed = sideSpeed;
    high_cmd.rotateSpeed = rotateSpeed;
    high_cmd.bodyHeight = bodyHeight;

    high_cmd.mode = mode;      // 0:idle, default stand      1:forced stand     2:walk continuously
    high_cmd.roll  = 0;
    high_cmd.pitch = 0;
    high_cmd.yaw = 0;
    udp.SetSend(high_cmd);
    udp.Send();

}

namespace py = pybind11;

// TODO: Expose all of comm.h and the RobotInterface Class.

PYBIND11_MODULE(robot_interface, m) {
  m.doc() = R"pbdoc(
          A1 Robot Interface Python Bindings
          -----------------------
          .. currentmodule:: a1_robot_interface
          .. autosummary::
             :toctree: _generate
      )pbdoc";

  py::class_<Cartesian>(m, "Cartesian")
      .def(py::init<>())
      .def_readwrite("x", &Cartesian::x)
      .def_readwrite("y", &Cartesian::y)
      .def_readwrite("z", &Cartesian::z);

  py::class_<IMU>(m, "IMU")
      .def(py::init<>())
      .def_readwrite("quaternion", &IMU::quaternion)
      .def_readwrite("gyroscope", &IMU::gyroscope)
      .def_readwrite("accelerometer", &IMU::accelerometer)
      .def_readwrite("rpy", &IMU::rpy)
      .def_readwrite("temperature", &IMU::temperature);

  py::class_<LED>(m, "LED")
      .def(py::init<>())
      .def_readwrite("r", &LED::r)
      .def_readwrite("g", &LED::g)
      .def_readwrite("b", &LED::b);

  py::class_<MotorState>(m, "MotorState")
      .def(py::init<>())
      .def_readwrite("mode", &MotorState::mode)
      .def_readwrite("q", &MotorState::q)
      .def_readwrite("dq", &MotorState::dq)
      .def_readwrite("ddq", &MotorState::ddq)
      .def_readwrite("tauEst", &MotorState::tauEst)
      .def_readwrite("q_raw", &MotorState::q_raw)
      .def_readwrite("dq_raw", &MotorState::dq_raw)
      .def_readwrite("ddq_raw", &MotorState::ddq_raw)
      .def_readwrite("temperature", &MotorState::temperature)
      .def_readwrite("reserve", &MotorState::reserve);

  py::class_<MotorCmd>(m, "MotorCmd")
      .def(py::init<>())
      .def_readwrite("mode", &MotorCmd::mode)
      .def_readwrite("q", &MotorCmd::q)
      .def_readwrite("dq", &MotorCmd::dq)
      .def_readwrite("tau", &MotorCmd::tau)
      .def_readwrite("Kp", &MotorCmd::Kp)
      .def_readwrite("Kd", &MotorCmd::Kd)
      .def_readwrite("reserve", &MotorCmd::reserve);

  py::class_<LowState>(m, "LowState")
      .def(py::init<>())
      .def_readwrite("levelFlag", &LowState::levelFlag)
      .def_readwrite("commVersion", &LowState::commVersion)
      .def_readwrite("robotID", &LowState::robotID)
      .def_readwrite("SN", &LowState::SN)
      .def_readwrite("bandWidth", &LowState::bandWidth)
      .def_readwrite("imu", &LowState::imu)
      .def_readwrite("motorState", &LowState::motorState)
      .def_readwrite("footForce", &LowState::footForce)
      .def_readwrite("footForceEst", &LowState::footForceEst)
      .def_readwrite("tick", &LowState::tick)
      .def_readwrite("wirelessRemote", &LowState::wirelessRemote)
      .def_readwrite("reserve", &LowState::reserve)
      .def_readwrite("crc", &LowState::crc);

  py::class_<LowCmd>(m, "LowCmd")
      .def(py::init<>())
      .def_readwrite("levelFlag", &LowCmd::levelFlag)
      .def_readwrite("commVersion", &LowCmd::commVersion)
      .def_readwrite("robotID", &LowCmd::robotID)
      .def_readwrite("SN", &LowCmd::SN)
      .def_readwrite("bandWidth", &LowCmd::bandWidth)
      .def_readwrite("motorCmd", &LowCmd::motorCmd)
      .def_readwrite("led", &LowCmd::led)
      .def_readwrite("wirelessRemote", &LowCmd::wirelessRemote)
      .def_readwrite("reserve", &LowCmd::reserve)
      .def_readwrite("crc", &LowCmd::crc);

  py::class_<HighState>(m, "HighState")
      .def(py::init<>())
      .def_readwrite("levelFlag", &HighState::levelFlag)
      .def_readwrite("commVersion", &HighState::commVersion)
      .def_readwrite("robotID", &HighState::robotID)
      .def_readwrite("SN", &HighState::SN)
      .def_readwrite("bandWidth", &HighState::bandWidth)
      .def_readwrite("mode", &HighState::mode)
      .def_readwrite("imu", &HighState::imu)
      .def_readwrite("forwardSpeed", &HighState::forwardSpeed)
      .def_readwrite("sideSpeed", &HighState::sideSpeed)
      .def_readwrite("rotateSpeed", &HighState::rotateSpeed)
      .def_readwrite("bodyHeight", &HighState::bodyHeight)
      .def_readwrite("updownSpeed", &HighState::updownSpeed)
      .def_readwrite("forwardPosition", &HighState::forwardPosition)
      .def_readwrite("sidePosition", &HighState::sidePosition)
      .def_readwrite("footPosition2Body", &HighState::footPosition2Body)
      .def_readwrite("footSpeed2Body", &HighState::footSpeed2Body)
      .def_readwrite("footForce", &HighState::footForce)
      .def_readwrite("footForceEst", &HighState::footForceEst)
      .def_readwrite("tick", &HighState::tick)
      .def_readwrite("wirelessRemote", &HighState::wirelessRemote)
      .def_readwrite("reserve", &HighState::reserve)
      .def_readwrite("crc", &HighState::crc);

  py::class_<HighCmd>(m, "HighCmd")
      .def(py::init<>())
      .def_readwrite("levelFlag", &HighCmd::levelFlag)
      .def_readwrite("commVersion", &HighCmd::commVersion)
      .def_readwrite("robotID", &HighCmd::robotID)
      .def_readwrite("SN", &HighCmd::SN)
      .def_readwrite("bandWidth", &HighCmd::bandWidth)
      .def_readwrite("mode", &HighCmd::mode)
      .def_readwrite("forwardSpeed", &HighCmd::forwardSpeed)
      .def_readwrite("sideSpeed", &HighCmd::sideSpeed)
      .def_readwrite("rotateSpeed", &HighCmd::rotateSpeed)
      .def_readwrite("bodyHeight", &HighCmd::bodyHeight)
      .def_readwrite("footRaiseHeight", &HighCmd::footRaiseHeight)
      .def_readwrite("yaw", &HighCmd::yaw)
      .def_readwrite("pitch", &HighCmd::pitch)
      .def_readwrite("roll", &HighCmd::roll)
      .def_readwrite("led", &HighCmd::led)
      .def_readwrite("wirelessRemote", &HighCmd::wirelessRemote)
      .def_readwrite("AppRemote", &HighCmd::AppRemote)
      .def_readwrite("reserve", &HighCmd::reserve)
      .def_readwrite("crc", &HighCmd::crc);

  py::class_<UDPState>(m, "UDPState")
      .def(py::init<>())
      .def_readwrite("TotalCount", &UDPState::TotalCount)
      .def_readwrite("SendCount", &UDPState::SendCount)
      .def_readwrite("RecvCount", &UDPState::RecvCount)
      .def_readwrite("SendError", &UDPState::SendError)
      .def_readwrite("FlagError", &UDPState::FlagError)
      .def_readwrite("RecvCRCError", &UDPState::RecvCRCError)
      .def_readwrite("RecvLoseError", &UDPState::RecvLoseError);

  py::class_<RobotInterface>(m, "RobotInterface")
      .def(py::init<uint8_t>())
      .def("receive_observation", &RobotInterface::ReceiveObservation)
      .def("send_command", &RobotInterface::SendCommand)
      .def("receive_high_observation", &RobotInterface::ReceiveHighObservation)
      .def("send_high_command", &RobotInterface::SendHighCommand);

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif

  m.attr("TEST") = py::int_(int(42));
}
