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

  public class IO {
    private IO() {

    }
    static IO io;
    public static IO GetInstance() {
      if (io == null)
        io = new IO();
      return io;
    }

    public bool[] X { get; } = new bool[32];
    public bool[] Y { get; } = new bool[32];

    //public delegate void OnMsgEventHandler(string msg);
    //public event OnMsgEventHandler OnMsgEvent;
    public event Action<string> OnMsgEvent;
    bool bRunThread;

    public bool BoardInit() {
      //NMCSDKLib.MC_STATUS ms = NMCSDKLib.MC_MasterInit(0);
      Status ms = NMC.MC_MasterInit(Global.BOARD_ID);
      if (ms != Status.MC_OK) {
        OnMsgEvent($"BoardInit fail : {ms}");
        return false;
      }
      OnMsgEvent($"BoardInit success");
      return true;
    }
    public bool BoardRun() {
      Status ms = NMC.MC_MasterRUN(Global.BOARD_ID);
      if (ms != Status.MC_OK) {
        OnMsgEvent($"BoardRun fail : {ms}");
        return false;
      }
      OnMsgEvent($"BoardRun success");
      return true;
    }
    public void On(int bit) {
      uint offset = (uint)bit / 8;
      byte bitOffset = (byte)(bit % 8);

      Status ms
        = NMC.MC_IO_WRITE_BIT(Global.BOARD_ID, Global.ECAT_ADDR, offset, bitOffset, true);
      if (ms != Status.MC_OK) {
        OnMsgEvent($"Bit On Fail : {ms}");
      }
      OnMsgEvent($"Bit On success");

    }
    public void Off(int bit) {
      uint offset = (uint)bit / 8;
      byte bitOffset = (byte)(bit % 8);

      Status ms
        = NMC.MC_IO_WRITE_BIT(Global.BOARD_ID, Global.ECAT_ADDR, offset, bitOffset, false);
      if (ms != Status.MC_OK) {
        OnMsgEvent($"Bit Off Fail : {ms}");
      }
      OnMsgEvent($"Bit Off success");

    }
    bool Update() {
      uint indata = 0, outdata = 0;
      Status ms
        = NMC.MC_IO_READ_DWORD(Global.BOARD_ID, Global.ECAT_ADDR, 1, 4, ref indata);

      if (ms != Status.MC_OK) {
        OnMsgEvent($"In IO Buf Read Fail : {ms}");
        return false;
      }

      ms = NMC.MC_IO_READ_DWORD(Global.BOARD_ID, Global.ECAT_ADDR, 0, 0, ref outdata);
      if (ms != Status.MC_OK) {
        OnMsgEvent($"Out IO Buf Read Fail : {ms}");
        return false;
      }

      for (int i = 0; i < Global.MAX_IO; i++) {
        int mask = 1 << i;
        X[i] = (indata & mask) != 0 ? true : false;
        Y[i] = (outdata & mask) != 0 ? true : false;
      }
      return true;
    }

    void Run() { //io update 메서드 호출 Loop
      try {
        OnMsgEvent("IO Update Loop 시작");
        while (bRunThread) {
          if (Update() == false) break;
          Thread.Sleep(100);
        }
      } 
      catch (Exception) { } 
      finally {
        OnMsgEvent("IO Update Loop 종료");
        bRunThread = false;
      }
    }

    public void Start() { //run method 스레드 시작
      if(bRunThread == true) {
        OnMsgEvent("IO Update Loop가 이미 실행 중입니다.");
        return;
      }
      Thread runThread = new Thread(Run);
      runThread.IsBackground = true; //main thread와 lifetime 동기화
      bRunThread = true;
      runThread.Start();
    }
  }
}
