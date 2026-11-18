using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Helper {
  public enum AxisItems {
    CmdVel, ActVel, CmdPos, ActPos, MAX
  }
  public enum AxisMap {
    ErrorStop,
    Disabled,
    Stopping,
    StandStill,
    DiscreteMotion,
    ContinuousMotion,
    SynchronizedMotion,
    Homing,
    SWLimitSwitchPosEvent,
    SWLimitSwitchNegEvent,
    ConstantVelocity,
    Accelerating,
    Decelerating,
    DirectionPositive,
    DirectionNegative,
    LimitSwitchNeg,
    LimitSwitchPos,
    HomeAbsSwitch,
    HWLimitSwitchPosEvent,
    HWLimitSwitchNegEvent,
    DriveFault,
    SensorStop,
    ReadyForPowerOn,
    PowerOn,
    IsHomed,
    AxisWarning,
    MotionComplete,
    Gearing,
    GroupMotion,
    BufferFull,
    Reserved,
    Debuglimitevent,
    MAX
  }
}
