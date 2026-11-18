using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NMCMotionSDK;
using Helper;
using System.Threading;

namespace MotionKit {
  using Status = NMCSDKLib.MC_STATUS;
  using NMC = NMCSDKLib;

  public class ServoInfo {
    //명사: 축번호, 속도지령, 실제속도, 위치지령, 실제위치, 32점의 상태값
    //동사: Update, Run(루프), Start(루프를 스레드 시작)
    public ServoInfo(ushort _AxisNum) {
      AxisNum = _AxisNum;
      status = new bool[(int)AxisMap.MAX];
      Start();
    }

    ushort AxisNum;
    double cmdVel, actVel, cmdPos, actPos;
    bool[] status;
    bool bRunThread;
    Thread runThread;

    public double CmdVel { get { return cmdVel; } }
    public double ActVel { get { return actVel; } }
    public double CmdPos { get { return cmdPos; } }
    public double ActPos { get { return actPos; } }

    //AxisStatus로 인덱서 작성
    public bool this[AxisMap idx] {
      get { return status[(int)idx]; }
    }

    bool Update() {
      //4개의 AxisItem에 대한 값과 32개의 AxisStatus를 읽어오고
      uint data = 0;

      Status[] errs = new Status[5];
      errs[0] = NMC.MC_ReadCommandedVelocity(Global.BOARD_ID, AxisNum, ref cmdVel);
      errs[1] = NMC.MC_ReadActualVelocity(Global.BOARD_ID, AxisNum, ref actVel);
      errs[2] = NMC.MC_ReadCommandedPosition(Global.BOARD_ID, AxisNum, ref cmdPos);
      errs[3] = NMC.MC_ReadActualPosition(Global.BOARD_ID, AxisNum, ref actPos);
      errs[4] = NMC.MC_ReadAxisStatus(Global.BOARD_ID, AxisNum, ref data);

      foreach (var item in errs) {
        if (item != Status.MC_OK)
          return false;
      }

      for (int i = 0; i < (int)AxisMap.MAX; i++) {
        int mask = 1 << i;
        status[i] = (data & mask) != 0 ? true : false;
      }
      return true;
    }
    void Run() {
      //루프
      try {
        while (bRunThread) {
          Thread.Sleep(100);
          if (Update() == false)
            break;
        }
      } catch (Exception) {
      } finally {
        bRunThread = false;
      }
    }
    void Start() {
      //루프 스레드 스타트
      if (bRunThread) return;
      bRunThread = true;

      runThread = new Thread(Run);
      runThread.IsBackground = true;
      runThread.Start();
    }
  }
}
