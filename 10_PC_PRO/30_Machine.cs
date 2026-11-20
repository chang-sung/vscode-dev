using Helper;
using NMCMotionSDK;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Net.NetworkInformation;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MotionKit {
  using NMC = NMCSDKLib;
  using Status = NMCSDKLib.MC_STATUS;

  public class Machine {
    //명사: x축, y축, 픽업실린더, 배큠, 테이블
    public Machine() {

      io = IO.GetInstance();

      XAxis = new Servo(Global.X_AXIS_NAME, Global.X_AXIS_NUM);
      YAxis = new Servo(Global.Y_AXIS_NAME, Global.Y_AXIS_NUM);
      Teach = new TeachData("MainTeach");
      // 공급실린더: 복솔
      LoadCyl = new Cylinder(IOMapY.LoadCylFwd, IOMapY.LoadCylBwd, IOMapX.LoadCylFwd, IOMapX.LoadCylBwd);
      // 편솔실린더:
      PickCyl = new Cylinder(IOMapY.PickCylDn, IOMapX.PickCylDn, IOMapX.PickCylUp);
      WorkCyl = new Cylinder(IOMapY.WorkCylDn , IOMapX.WorkCylDn, IOMapX.WorkCylUp);
      MoveCyl = new Cylinder(IOMapY.MoveCylFwd, IOMapX.MoveCylFwd, IOMapX.MoveCylBwd);
      UnldCyl = new Cylinder(IOMapY.UnldCyleFwd, IOMapX.UnldCyleFwd, IOMapX.UnldCyleBwd);
      PickVac = new Vacuum(IOMapY.VacOn, IOMapY.VacOff, IOMapX.PickVacSen2);

      // Lamp
      LampRed = new Lamp(IOMapY.LampRed);
      LampYellow = new Lamp(IOMapY.LampYellow);
      LampGreen = new Lamp(IOMapY.LampGreen);

      // Motor 
      Drill = new Actuator(IOMapY.Drill);
      Convy = new Actuator(IOMapY.Conveyor);

      drillTimer = new Stopwatch();
      convTimer = new Stopwatch();
   
      cmd = EqpCmd.None;
      workStep = inspStep = AutoStep.Ready;
      drillTimer = new Stopwatch();
      convTimer = new Stopwatch();
      bWorkExist = bInspExist = false;

    }

    public Servo XAxis { get; }
    public Servo YAxis { get; }
    public TeachData Teach { get; }
    public IActuator PickCyl { get; }
    public IActuator PickVac { get; }
    public IActuator LoadCyl { get; }
    public IActuator WorkCyl {  get; }
    public IActuator MoveCyl { get; }
    public IActuator UnldCyl { get; }
    public IActuator Drill { get; }
    public IActuator Convy { get; }

    public Lamp LampRed { get; }
    public Lamp LampYellow { get; }
    public Lamp LampGreen { get; }

    public event Action<string> OnMsgEvent;

    IO io;


    Thread runThread;
    bool bRunThread = false;
    Inspector inspector;
    AutoStep workStep, inspStep;
    EqpCmd cmd;
    private EqpStatus currentStatus = EqpStatus.Stop;


    Stopwatch drillTimer, convTimer;
    bool bWorkExist, bInspExist;
    
    public event Action<EqpStatus> EqpStatusChanged;
    public Func<bool> SearchColor { get; set; }

    public EqpStatus eqpStatus
    {
        get { return currentStatus; }
    }

    private EqpStatus status
    {
        get { return currentStatus; }
        set
        {
            if (currentStatus == value)
                return;

            currentStatus = value;
            LampUpdate(value);
            EqpStatusChanged?.Invoke(value);
        }
    }



    void Run() {
      try {
        while (bRunThread) {
          io.Update();
          EQPRun();                    //14
          Thread.Sleep(100);
        }
      }
      catch (Exception e) {
        Console.WriteLine("Exception : " + e.Message);
      }
      finally {
        bRunThread = false;
      }
    }


        //IO Update하는 thread
    public bool Create()
    {
        if (bRunThread)
            return true;

        bRunThread = true;

        runThread = new Thread(Run);
        runThread.IsBackground = true;
        runThread.Start();

        return true;
    }


    //시퀀스제어 이벤트핸들러
    //Thread 돌릴 부분으로 EqpStatus/EqpCmd의 설비상태 관리
    void EQPRun() {
      switch (cmd) {
        case EqpCmd.None: break;
        case EqpCmd.Init:
          //뭔가 실제 이닛 동작을 하는 메서드를 호출해주고
          //그게 성공하면 status를 Ready로 바꿔주고
          //cmd는 이제 None으로 바꿔주면 됩니다.
          //=======> Init은 그냥 CmdInit() 에서 동작 시켜줄게요.
          //여기선 status/cmd만 재정의 해주겠습니다.
          status = EqpStatus.Ready;
          cmd = EqpCmd.None;
          break;
        case EqpCmd.EStop:
          status = EqpStatus.Error;
          cmd = EqpCmd.None;
          break;
        case EqpCmd.OneCycleStart:
        case EqpCmd.ContiCycleStop:
        case EqpCmd.CycleStop:
          WorkSEQ();
          InspSeq();
          break;
      }
    }

    public bool CmdInit() {
      if (eqpStatus == EqpStatus.Ready) {
        return false;
      }

      if (eqpStatus == EqpStatus.Run) {
        return false;
      }

      if (eqpStatus == EqpStatus.Idle) {
        return false;
      }

      LoadCyl.Off();
      WorkCyl.Off();
      MoveCyl.Off();
      UnldCyl.Off();

      Drill.Off();
      Convy.Off();

      workStep = inspStep = AutoStep.Ready;
      bWorkExist = bInspExist = false;

      cmd = EqpCmd.Init;
      return true;
    }

    public bool CmdEStop() {
      Drill.Off();
      Convy.Off();
      cmd = EqpCmd.EStop;
      return true;
    }

    public bool CmdOneCycleStart() {
      if (status != EqpStatus.Ready)          //Ready 상태에서만 One Cycle 가능
      {
        return false;
      }

      workStep = inspStep = AutoStep.step00;      //시퀀스 초기화
      cmd = EqpCmd.OneCycleStart;
      return true;
    }

    public bool CmdContCycleStart() {
      if (status != EqpStatus.Ready)      //Ready 상태에서만 Conti Cycle 가능
      {
        return false;
      }

      workStep = inspStep = AutoStep.step00;
      cmd = EqpCmd.ContiCycleStop;
      return true;
    }

    public bool CmdCycleStop()           //Idle 또는 Run 중에만 CycleStop 가능
    {
      if (!(status == EqpStatus.Idle || status == EqpStatus.Run)) {
        return false;
      }

      cmd = EqpCmd.CycleStop;
      return true;
    }




    /* 참고사항 : 관리중인 인덱스들
     * workStep/inspStep
     * bWorkExist/bInspExist
     * AutoStep.Ready / ... / Complete
     * EqpCmd cmd
     * EpqStatus status
     * more... 새로 만들어도 됨.
     */

    Stopwatch sw;
    void WorkSEQ()   //Loading 단계 sequence : 완성도 85%
    {
      switch (workStep) {
        case AutoStep.Ready://대기
          workStep = AutoStep.step00;
          break;
        case AutoStep.step00://로드실 전진, status run
          if (io.X[(int)IOMapX.MagazineSen] == false)         //io.X[12], 매거진 검출 센서
          {
            status = EqpStatus.Idle;
            workStep = AutoStep.Ready;
          }
          else {
            status = EqpStatus.Run;
            LoadCyl.On();
            LoadCyl.IsOn();
            workStep = AutoStep.step10;
            Thread.Sleep(500);
          }
          break;
        case AutoStep.step10://로드실 후진, bWorkExist true					
          LoadCyl.Off();
          bWorkExist = true;
          workStep = AutoStep.step20;
          Thread.Sleep(500);
          break;
        case AutoStep.step20://워크실 하강
          WorkCyl.On();
          workStep = AutoStep.step30;
          Thread.Sleep(500);
          break;
        case AutoStep.step30://드릴온 (stopwatch 시작)					
          sw = new Stopwatch();
          sw.Start();
          Drill.On();
          workStep = AutoStep.step40;
          break;
        case AutoStep.step40://가공시간 확인하여 드릴오프, 워크실 상승					
          if (sw.ElapsedMilliseconds > 1000) {
            sw.Stop();
            Drill.Off();
            WorkCyl.Off();
            workStep = AutoStep.step50;
          }
          break;
        case AutoStep.step50://bTestExist확인하여 무브실 전진
          MoveCyl.On();
          workStep = AutoStep.step60;
          Thread.Sleep(1000);
          break;
        case AutoStep.step60://무브실 후진, bWorkExist false, bInspExist true					
          MoveCyl.Off();
          bWorkExist = false;          //work 시퀀스 false = off
          bInspExist = true;          //Inspection 시퀀스 true
          workStep = AutoStep.step70;
          break;
        case AutoStep.step70: //컴플릿
                              //Conv.On();
          workStep = AutoStep.Complete;
          break;
        case AutoStep.Complete: //cmd확인하여 step을 Ready/step00 분기
                                //if (cmd == EqpCmd.ContiCycleStop)
                                //{
                                //	workStep = AutoStep.Ready;
                                //}
                                //else
                                //{
                                //	workStep = AutoStep.Ready;

          //}

          // OneCycleStart는 검사까지 완료될 때까지 이 단계에서 대기한다.
          if (cmd == EqpCmd.OneCycleStart)
            break;

          if (cmd == EqpCmd.CycleStop)
            workStep = AutoStep.Ready;
          else
            workStep = AutoStep.step00;

          break;


      }
    }

    Stopwatch swConv;


    void InspSeq()   //검사 단계 sequence : 완성도 85%
    {
      switch (inspStep) {
        case AutoStep.Ready:
          inspStep = AutoStep.step00;

          break;
        case AutoStep.step00:
          if (bInspExist == true) {
            swConv = new Stopwatch();
            Convy.On();
            swConv.Start();
            inspStep = AutoStep.step10;
            bInspExist = false;
            status = EqpStatus.Run;
          }
          else {
            inspStep = AutoStep.Ready;
          }
          break;
          case AutoStep.step10:
            if (swConv.ElapsedMilliseconds > Global.INSPECT_TIME)
            {
              Convy.Off();
              swConv.Stop();

              try
              {
                Func<bool> inspect = SearchColor;

                bool detected = inspect();

                inspStep = detected ? AutoStep.step200 : AutoStep.step100;

                OnMsg(detected
                  ? "색상 검출 → step200"
                  : "색상 미검출 → step100");
              }
              catch (Exception ex)
              {
                OnMsg("색상 검사 실패: " + ex.Message);
                cmd = EqpCmd.EStop;
              }
            }
            break;
        case AutoStep.step100:                  //conv on, convstopwatch 재시작
          Convy.On();
          swConv.Restart();
          inspStep = AutoStep.step110;
          break;
        case AutoStep.step110: //취출위치 도착하면 conv off하고 unldcyl 전진
          Console.WriteLine("time 1: " + DateTime.Now + " " + swConv);
          if (swConv.ElapsedMilliseconds > 300) {
            UnldCyl.On();
            Console.WriteLine("time 2: " + DateTime.Now + " " + swConv);

            swConv.Stop();    //컨베이어 생성 위치에 따라 Reset이 필요할 수 있음. --> Reset사용이 바람직.
            Thread.Sleep(2000);
            UnldCyl.Off();
            Thread.Sleep(1000);

            inspStep = AutoStep.step120;
          }
          Console.WriteLine("time 3: " + DateTime.Now + " " + swConv);
          break;
        case AutoStep.step120: //unldcyl 전진 확인하여  unldcyl 후진
          Thread.Sleep(1000);
          UnldCyl.Off();
          inspStep = AutoStep.step130;
          break;
        case AutoStep.step130: //unldcyl 후진 확인하여 컴플릿
          if (UnldCyl.IsOff() == true)     //io.X[6], unldcyl이 후진이면
          {
            inspStep = AutoStep.Complete;
          }
          break;
        case AutoStep.step200: // 비금속(양품) // conv on
          Convy.On();
          inspStep = AutoStep.step210;
          break;
        case AutoStep.step210: //ENDSEN 감지되면 CONV OFF
          if (io.X[(int)IOMapX.ConveyorEndSen] == true)           //io.X[16]
          {
            Thread.Sleep(300);
            Convy.Off();
            inspStep = AutoStep.step220;
          }
          break;
        case AutoStep.step220: //ENDSEN 감지 해제되면 컴플릿 
          if (io.X[(int)IOMapX.ConveyorEndSen] == false) {
            inspStep = AutoStep.step230;
          }
          break;
        case AutoStep.step230: //CMD 확인하여 CYCLESTOP, ONE CYCLE START면 Ready Step으로 (STATUS = READY로 변경)
                               //CONTICYCLESTART면, STEP00으로(재공과 투입된 WORK가 없으면 STATUS=IDLE로)
          if (cmd == EqpCmd.ContiCycleStop) {
            inspStep = AutoStep.Complete;
          }
          else if (cmd == EqpCmd.OneCycleStart) {
            FinishOneCycle();
          }
          else if (cmd == EqpCmd.CycleStop) {
            inspStep = AutoStep.Ready;
            status = EqpStatus.Stop;
          }
          break;
        case AutoStep.Complete:
          if (cmd == EqpCmd.OneCycleStart) {
            FinishOneCycle();
          }
          else {
            inspStep = AutoStep.Ready;
            status = EqpStatus.Idle;
            Convy.Off();
          }
          break;
      }

    }

    // 작업과 검사가 모두 끝난 뒤에만 OneCycle 명령을 종료한다.
    void FinishOneCycle() {
      Convy.Off();
      workStep = AutoStep.Ready;
      inspStep = AutoStep.Ready;
      cmd = EqpCmd.None;
      status = EqpStatus.Ready;
    }

    void OnMsg(string msg) {
      if (OnMsgEvent != null) OnMsgEvent(msg);
    }

    public bool MultMoveAbs(TeachList pos) {
      var axisNums = new ushort[] { Global.X_AXIS_NUM, Global.Y_AXIS_NUM };
      var axisPos = new double[] { Teach[pos].X, Teach[pos].Y };
      var axisDir = new NMC.MC_DIRECTION[] {
                NMC.MC_DIRECTION.mcCurrentDirection, NMC.MC_DIRECTION.mcCurrentDirection };

      Status ms = NMC.MC_MoveAbsoluteMultiAxis(Global.BOARD_ID, 2, axisNums, axisPos,
          XAxis.Param[ParamList.AbsVel],
          XAxis.Param[ParamList.AbsAcc],
          XAxis.Param[ParamList.AbsDec],
          XAxis.Param[ParamList.AbsJerk],
          axisDir, 1);
      if (ms != Status.MC_OK) {
        OnMsg($"{pos} MultMoveAbs Error : {ms}");
        return false;
      }
      OnMsg($"{pos} MultMoveAbs Start");
      return true;
    }

    public bool Home() {
      // 진공 Off,
      // 실린더 상승(off)
      // x, y 축 home
      // 초기화 완료 여부 확인
      OnMsg("장비 초기화 시작");
      PickVac.Off();
      PickCyl.Off();

      if (XAxis.ServoHome() == false || YAxis.ServoHome() == false)
        return false;

      Thread.Sleep(1000);
      int timeout = 0;
      while (true) {
        Thread.Sleep(100);
        if (PickVac.IsOff() && PickCyl.IsOff()
            && XAxis.Info[AxisMap.IsHomed] && YAxis.Info[AxisMap.IsHomed])
          break;
        if (timeout++ > Global.HOME_TIMEOUT) {
          OnMsg($"장비 초기화 TimeOut : {Global.HOME_TIMEOUT / 10} 초");
          return false;
        }
      }
      OnMsg("장비 초기화 완료");
      return true;
    }

    public bool WorkMove(TeachList from, TeachList to) {
      int autoStep = 0, oldStep = -1, timeOut = 0;
      // 초기화 완료 여부
      if (!(PickVac.IsOff() && PickCyl.IsOff()
          && XAxis.Info[AxisMap.IsHomed] && YAxis.Info[AxisMap.IsHomed])) {
        OnMsg("장비 초기화 상태를 확인 하세요");
        return false;
      }

      while (true) {
        Thread.Sleep(100);

        // step간 100이상 정체시 timeout
        if (autoStep != oldStep) {
          timeOut = 0;
          oldStep = autoStep;
        }
        else {
          if (timeOut++ > 100) {
            OnMsg($"{autoStep} Step Time Out Error : {timeOut}");
            return false;
          }
        }

        switch (autoStep) {
          case 0:
            MultMoveAbs(from);
            Thread.Sleep(200);
            autoStep++;
            break;
          case 1:
            // from 이동 완료 체크
            //xAxis.Info[AxisMap.MotionComplete] 이동 완료
            if (XAxis.Info[AxisMap.MotionComplete] && YAxis.Info[AxisMap.MotionComplete]) {
              // 실린더 하강
              PickCyl.On();
              autoStep++;
            }
            break;
          case 2:
            // 실린더 하강 확인
            if (PickCyl.IsOn()) {
              // vac on
              PickVac.On();
              autoStep++;
            }
            break;
          case 3:
            // vac on 확인
            if (PickVac.IsOn()) {
              // 실린더 상승
              PickCyl.Off();
              autoStep++;
            }
            break;
          case 4:
            // 실린더 상승 확인
            if (PickCyl.IsOff()) {
              // to로 이동
              MultMoveAbs(to);
              Thread.Sleep(200);
              autoStep++;
            }
            break;
          case 5:
            // to 이동 확인
            if (XAxis.Info[AxisMap.MotionComplete] && YAxis.Info[AxisMap.MotionComplete]) {
              // 실린더 하강
              PickCyl.On();
              autoStep++;
            }
            break;
          case 6:
            // 실린더 하강확인
            if (PickCyl.IsOn()) {
              // vac off
              PickVac.Off();
              autoStep++;
            }
            break;
          case 7:
            // vac off 확인
            if (PickVac.IsOff()) {
              // 시린더 상승
              PickCyl.Off();
              autoStep++;
            }
            break;
          case 8:
            if (PickCyl.IsOff()) {
              // 시린더 상승 확인
              return true;
            }
            break;
        }
      }
    }


    public void LampUpdate(EqpStatus status) {
      switch (status) {
        case EqpStatus.Ready:
          LampRed.On(); LampYellow.On(); LampGreen.On();
          break;
        case EqpStatus.Stop:
          LampRed.Off(); LampYellow.Off(); LampGreen.Off();
          break;
        case EqpStatus.Error:
          LampRed.On(); LampYellow.Off(); LampGreen.Off();
          break;
        case EqpStatus.Run:
          LampRed.Off(); LampYellow.Off(); LampGreen.On();
          break;
        case EqpStatus.Idle:
          LampRed.Off(); LampYellow.On(); LampGreen.Off();
          break;
        default:
          break;
      }
    }









  }
}
