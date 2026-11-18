using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NMCMotionSDK;
using Helper;

namespace MotionKit {
  using Status = NMCSDKLib.MC_STATUS;
  using NMC = NMCSDKLib;

  public class Servo {
    //명사: 축이름, 축번호, 모션파람, 모션정보(지령값, 현재값, 센서)
    //동사: 서보on/off, reset, e-stop, 초기화, 메세지 이벤트
    //      절대위치이동, 상대위치이동, 속도이동
    public Servo(string _axisName, ushort _axisNum) {
      axisName = _axisName;
      axisNum = _axisNum;
      Info = new ServoInfo(_axisNum);
      //Param = new ServoParam(Global.ParamPath + axisName + ".txt");
      Param = new ServoParam(axisName);
    }

    string axisName;
    ushort axisNum;
    public ServoInfo Info { get; }
    public ServoParam Param { get; }

    public event Action<string> OnMsgEvent;
    public void OnMsg(string msg) {
      if (OnMsgEvent != null)
        OnMsgEvent(msg);
    }

    public bool ServoOn() {
      Status ms = NMC.MC_Power(Global.BOARD_ID, axisNum, true);
      if (ms != Status.MC_OK) {
        //메세지 띄우고
        OnMsg($"{axisName} : ServoOn Fail {ms}");
        return false;
      }
      OnMsg($"{axisName} : ServoOn Success {ms}");
      return true;
    }
    public bool ServoOff() {
      Status ms = NMC.MC_Power(Global.BOARD_ID, axisNum, false);
      if (ms != Status.MC_OK) {
        OnMsg($"{axisName} : ServoOff Fail {ms}");
        return false;
      }
      OnMsg($"{axisName} : ServoOff Success {ms}");
      return true;
    }
    public bool ServoStop() {
      //Status ms = NMC.MC_Stop(Global.BOARD_ID, axisNum, true, 50000, 500000);
      Status ms = NMC.MC_Stop(Global.BOARD_ID, axisNum, true, Param[ParamList.JogDec], Param[ParamList.JogJerk]);
      if (ms != Status.MC_OK) {
        OnMsg($"{axisName} : ServoStop Fail {ms}");
        return false;
      }
      OnMsg($"{axisName} : ServoStop Success {ms}");
      return true;
    }
    public bool ServoReset() {
      Status ms = NMC.MC_Reset(Global.BOARD_ID, axisNum);
      if (ms != Status.MC_OK) {
        OnMsg($"{axisName} : ServoReset Fail {ms}");
        return false;
      }
      OnMsg($"{axisName} : ServoReset Success {ms}");
      return true;
    }
    public bool ServoHome() {
      Status ms = NMC.MC_Home(Global.BOARD_ID, axisNum, 0, NMC.MC_BUFFER_MODE.mcAborting);
      if (ms != Status.MC_OK) {
        OnMsg($"{axisName} : ServoHome Fail {ms}");
        return false;
      }
      OnMsg($"{axisName} : ServoHome Start {ms}");
      return true;
    }
    public bool ServoHalt() {
      Status ms = NMC.MC_Halt(Global.BOARD_ID, axisNum, Param[ParamList.JogDec], Param[ParamList.JogJerk], NMC.MC_BUFFER_MODE.mcAborting);
      if (ms != Status.MC_OK) {
        OnMsg($"{axisName} : ServoHalt Fail {ms}");
        return false;
      }
      OnMsg($"{axisName} : ServoHalt Success {ms}");
      return true;
    }
    public bool MoveVelNeg() {
      Status ms = NMC.MC_MoveVelocity(Global.BOARD_ID, axisNum, 
        Param[ParamList.JogVel], Param[ParamList.JogAcc], Param[ParamList.JogDec], Param[ParamList.JogJerk], 
        NMC.MC_DIRECTION.mcNegativeDirection, NMC.MC_BUFFER_MODE.mcAborting);
      if (ms != Status.MC_OK) {
        OnMsg($"{axisName} : MoveVelNeg Fail {ms}");
        return false;
      }
      OnMsg($"{axisName} : MoveVelNeg Start {ms}");
      return true;
    }
    public bool MoveVelPos() {
      Status ms = NMC.MC_MoveVelocity(Global.BOARD_ID, axisNum, 
        Param[ParamList.JogVel], Param[ParamList.JogAcc], Param[ParamList.JogDec], Param[ParamList.JogJerk],
        NMC.MC_DIRECTION.mcPositiveDirection, NMC.MC_BUFFER_MODE.mcAborting);
      if (ms != Status.MC_OK) {
        OnMsg($"{axisName} : MoveVelPos Fail {ms}");
        return false;
      }
      OnMsg($"{axisName} : MoveVelPos Start {ms}");
      return true;
    }
    public bool MoveRel(double dist_mm) {
      double dist = dist_mm * 1000;
      Status ms = NMC.MC_MoveRelative(Global.BOARD_ID, axisNum, dist,
                                      Param[ParamList.RelVel],
                                      Param[ParamList.RelAcc],
                                      Param[ParamList.RelDec],
                                      Param[ParamList.RelJerk],
                                      NMC.MC_BUFFER_MODE.mcAborting);
      if (ms != Status.MC_OK) {
        OnMsg($"{axisName} MoveRel Error : {ms}");
        return false;
      }
      OnMsg($"{axisName} MoveRel Start");
      return true;
    }
  }
}
