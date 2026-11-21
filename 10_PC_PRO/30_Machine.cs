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
using System.Windows.Forms;

namespace MotionKit
{
  using NMC = NMCSDKLib;
  using Status = NMCSDKLib.MC_STATUS;

  public class Machine
  {
    //명사: x축, y축, 픽업실린더, 배큠, 테이블
    public Machine()
    {

      io = IO.GetInstance();

      XAxis = new Servo(Global.X_AXIS_NAME, Global.X_AXIS_NUM);
      YAxis = new Servo(Global.Y_AXIS_NAME, Global.Y_AXIS_NUM);
      Teach = new TeachData("MainTeach");
      // 공급실린더: 복솔
      LoadCyl = new Cylinder(IOMapY.LoadCylFwd, IOMapY.LoadCylBwd, IOMapX.LoadCylFwd, IOMapX.LoadCylBwd);
      // 편솔실린더:
      PickCyl = new Cylinder(IOMapY.PickCylDn, IOMapX.PickCylDn, IOMapX.PickCylUp);
      WorkCyl = new Cylinder(IOMapY.WorkCylDn, IOMapX.WorkCylDn, IOMapX.WorkCylUp);
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
      workStep = inspStep = unldStep = AutoStep.Ready;
      drillTimer = new Stopwatch();
      convTimer = new Stopwatch();
      bWorkExist = bInspExist = bUnldExist = false;

    }

    public Servo XAxis { get; }
    public Servo YAxis { get; }
    public TeachData Teach { get; }
    public IActuator PickCyl { get; }
    public IActuator PickVac { get; }
    public IActuator LoadCyl { get; }
    public IActuator WorkCyl { get; }
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
    bool OneCycle_WorkStepEnd = false;
    bool OneCycle_InspStepEnd = false;
    bool OneCycle_UnldStepEnd = false;

    AutoStep workStep, inspStep, unldStep;
    volatile EqpCmd cmd;
    bool sequenceFault;
    volatile EqpStatus currentStatus = EqpStatus.Stop;


    Stopwatch drillTimer, convTimer;
    bool bWorkExist, bInspExist, bUnldExist;

    public event Action<EqpStatus> EqpStatusChanged;
    public delegate bool SearchColorHandler(out Bitmap cropImage);
    public SearchColorHandler SearchColor { get; set; }
    public event Action<Bitmap, bool> InspectionCompleted;

    private int productOkCount;
    private int productNgCount;

    public int ProductOkCount => System.Threading.Volatile.Read(ref productOkCount);
    public int ProductNgCount => System.Threading.Volatile.Read(ref productNgCount);

    private DateTime productInputTime;
    private string productInspectionResult;
    private int nextUnloadIndex;

    public event Action<DateTime, string, string> ProductUnloadPositionSelected;


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

    void Run()
    {
      try
      {
        while (bRunThread)
        {
          io.Update();
          EQPRun();                    //14
          Thread.Sleep(100);
        }
      }
      catch (Exception e)
      {
        Console.WriteLine("Exception : " + e.Message);
      }
      finally
      {
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
    public bool CanStartCycle
    {
      get { return status == EqpStatus.Ready && cmd == EqpCmd.None; }
    }

    void EQPRun()
    {
      switch (cmd)
      {
        case EqpCmd.Init:
          if (XAxis.Info[AxisMap.MotionComplete] &&
              YAxis.Info[AxisMap.MotionComplete])
          {
            cmd = EqpCmd.None;
            status = EqpStatus.Ready;
          }
          break;
        case EqpCmd.EStop:
          cmd = EqpCmd.None;
          status = EqpStatus.Stop;
          break;
        case EqpCmd.OneCycleStart:
        case EqpCmd.ContiCycleStop:
        case EqpCmd.CycleStop:
          WorkSEQ();
          if (cmd == EqpCmd.EStop)
            return;

          InspSeq();
          if (HandleSequenceFault() || cmd == EqpCmd.EStop)
            return;

          UnldSeq();
          if (HandleSequenceFault() || cmd == EqpCmd.EStop)
            return;

          UpdateCycleStatus();
          break;
      }
    }

    private bool HandleSequenceFault()
    {
      if (!sequenceFault)
        return false;

      sequenceFault = false;
      Drill.Off();
      Convy.Off();

      if (cmd != EqpCmd.EStop)
      {
        cmd = EqpCmd.None;
        status = EqpStatus.Error;
      }

      return true;
    }

    private void UpdateCycleStatus()
    {
      EqpCmd activeCommand = cmd;
      if (activeCommand != EqpCmd.OneCycleStart &&
          activeCommand != EqpCmd.ContiCycleStop &&
          activeCommand != EqpCmd.CycleStop)
        return;

      bool productInProcess = bWorkExist || bInspExist || bUnldExist ||
          IsSequenceRunning(workStep) ||
          IsSequenceRunning(inspStep) ||
          IsSequenceRunning(unldStep);

      bool allSequencesSettled = workStep != AutoStep.step00 &&
          inspStep != AutoStep.step00 &&
          unldStep != AutoStep.step00;

      // NG 제품은 검사 시퀀스의 배출 분기에서 작업이 끝난다.
      bool oneCycleFinished = activeCommand == EqpCmd.OneCycleStart &&
          OneCycle_WorkStepEnd && OneCycle_InspStepEnd;

      if (!productInProcess && allSequencesSettled &&
          (activeCommand == EqpCmd.CycleStop || oneCycleFinished))
      {
        cmd = EqpCmd.None;
        status = EqpStatus.Ready;
        return;
      }

      status = productInProcess ? EqpStatus.Run : EqpStatus.Idle;
    }

    private static bool IsSequenceRunning(AutoStep step)
    {
      return step != AutoStep.Ready &&
          step != AutoStep.step00 &&
          step != AutoStep.Complete;
    }

    public bool CmdInit()
    {
      if (cmd != EqpCmd.None ||
          (status != EqpStatus.Stop && status != EqpStatus.Error))
        return false;

      LoadCyl.Off();
      WorkCyl.Off();
      MoveCyl.Off();
      UnldCyl.Off();
      PickCyl.Off();
      PickVac.Off();

      Drill.Off();
      Convy.Off();

      if (!MultMoveAbs(TeachList.Pos_LD))
      {
        status = EqpStatus.Error;
        return false;
      }
      workStep = inspStep = unldStep = AutoStep.Ready;
      bWorkExist = bInspExist = bUnldExist = false;

      cmd = EqpCmd.Init;
      return true;
    }

    public bool CmdEStop()
    {
      Drill.Off();
      Convy.Off();
      cmd = EqpCmd.EStop;
      return true;
    }

    public bool CmdOneCycleStart()
    {
      if (!CanStartCycle)
        return false;

      workStep = inspStep = unldStep = AutoStep.step00;      //시퀀스 초기화
      OneCycle_InspStepEnd = OneCycle_UnldStepEnd = OneCycle_WorkStepEnd = false;
      cmd = EqpCmd.OneCycleStart;
      return true;
    }

    public bool CmdContCycleStart()
    {
      if (!CanStartCycle)
        return false;

      workStep = inspStep = unldStep = AutoStep.step00;
      OneCycle_InspStepEnd = OneCycle_UnldStepEnd = OneCycle_WorkStepEnd = false;
      cmd = EqpCmd.ContiCycleStop;
      return true;
    }

    public bool CmdCycleStop()           //Idle 또는 Run 중에만 CycleStop 가능
    {
      if ((cmd != EqpCmd.OneCycleStart && cmd != EqpCmd.ContiCycleStop) ||
          (status != EqpStatus.Idle && status != EqpStatus.Run))
      {
        return false;
      }

      cmd = EqpCmd.CycleStop;
      return true;
    }

    public void LampUpdate(EqpStatus status)
    {
      switch (status)
      {
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


    /* 참고사항 : 관리중인 인덱스들
     * workStep /inspStep / unldStep
     * bWorkExist/bInspExist
     * AutoStep.Ready / ... / Complete
     * EqpCmd cmd
     * EpqStatus status
     * more... 새로 만들어도 됨.
     */

    Stopwatch sw;
    void WorkSEQ()   //Loading 단계 sequence : 완성도 85%
    {
      switch (workStep)
      {
        case AutoStep.Ready://대기
          if (cmd != EqpCmd.CycleStop &&
              io.X[(int)IOMapX.MagazineSen] && !OneCycle_WorkStepEnd)
            workStep = AutoStep.step00;
          break;
        case AutoStep.step00://로드실 전진, status run
          if (cmd == EqpCmd.CycleStop || !io.X[(int)IOMapX.MagazineSen])         //io.X[12], 매거진 검출 센서
          {
            workStep = AutoStep.Ready;
          }
          else
          {

            LoadCyl.On();
            productInputTime = DateTime.Now;
            productInspectionResult = null;

            workStep = AutoStep.step10;
            Thread.Sleep(1000);

          }
          break;
        case AutoStep.step10://로드실 후진, bWorkExist true                 
          LoadCyl.Off();

          bWorkExist = true;
          workStep = AutoStep.step20;
          Thread.Sleep(1000);
          break;
        case AutoStep.step20://워크실 하강
          WorkCyl.On();

          workStep = AutoStep.step30;
          Thread.Sleep(1000);
          break;
        case AutoStep.step30://드릴온 (stopwatch 시작)                  
          sw = new Stopwatch();
          sw.Start();
          Drill.On();
          workStep = AutoStep.step40;
          break;
        case AutoStep.step40://가공시간 확인하여 드릴오프, 워크실 상승                  
          if (sw.ElapsedMilliseconds > 1000)
          {
            sw.Stop();
            Drill.Off();
            WorkCyl.Off();

            workStep = AutoStep.step50;
          }
          break;
        case AutoStep.step50://bTestExist확인하여 무브실 전진
          if (bInspExist == false && bUnldExist == false)
          {
            MoveCyl.On();
            Thread.Sleep(1000);
            workStep = AutoStep.step60;
          }
          break;
        case AutoStep.step60://무브실 후진, bWorkExist false, bInspExist true
          if (MoveCyl.IsOn())
          {
            MoveCyl.Off();
            bWorkExist = false;          //work 시퀀스 false = off
            bInspExist = true;          //Inspection 시퀀스 true
            Thread.Sleep(1000);
            workStep = AutoStep.step70;
          }

          break;
        case AutoStep.step70: //컴플릿
                              //Conv.On();
          workStep = AutoStep.Complete;
          break;
        case AutoStep.Complete: //cmd확인하여 step을 Ready/step00 분기

          if (cmd == EqpCmd.CycleStop || cmd == EqpCmd.OneCycleStart)
          {
            OneCycle_WorkStepEnd = true;
            workStep = AutoStep.Ready;
          }
          else
            workStep = AutoStep.step00;

          break;
      }
    }

    public Stopwatch swConv;

    public long ConvElapsedTime
    {
      get { return swConv?.ElapsedMilliseconds ?? 0; }
    }


    void InspSeq()   //검사 단계 sequence : 완성도 85%
    {
      switch (inspStep)
      {
        case AutoStep.Ready:
          if (bInspExist == true && bUnldExist == false && OneCycle_InspStepEnd == false)
          {
            inspStep = AutoStep.step00;
          }
          break;
        case AutoStep.step00:
          if (bInspExist == true && bUnldExist == false)
          {
            swConv = new Stopwatch();
            Convy.On();
            Thread.Sleep(100);
            swConv.Start();

            inspStep = AutoStep.step10;

          }
          else
          {
            inspStep = AutoStep.Ready;
          }
          break;
        case AutoStep.step10:
          if (swConv.ElapsedMilliseconds > 30)
          {
            Convy.Off();
            swConv.Stop();
            Thread.Sleep(100);

            try
            {

              var inspect = SearchColor;
              if (inspect == null)
                throw new InvalidOperationException("검사 메서드가 연결되지 않았습니다.");

              Bitmap cropImage;
              bool detected = inspect(out cropImage);
              productInspectionResult = detected ? "OK" : "NG";

              // 제품 1개의 검사 결과를 누적 집계
              if (detected)
              {
                Interlocked.Increment(ref productOkCount);
              }
              else
              {
                Interlocked.Increment(ref productNgCount);
              }

              using (cropImage)
              {
                inspStep = detected ? AutoStep.step200 : AutoStep.step100;
                InspectionCompleted?.Invoke(cropImage, detected);
              }

              Thread.Sleep(1000);

              OnMsg(detected
                  ? "OK → step200"
                  : "NG → step100");
            }
            catch (Exception ex)
            {
              OnMsg("검사 실패: " + ex.Message);
              sequenceFault = true;
            }
          }
          break;
        case AutoStep.step100:                  //conv on, convstopwatch 재시작
          Convy.On();
          Thread.Sleep(100);
          swConv.Restart();


          inspStep = AutoStep.step110;
          break;

        case AutoStep.step110:
          if (swConv.ElapsedMilliseconds > 1000)
          {
            Convy.Off();
            UnldCyl.On();

            Console.WriteLine("time 2: " + DateTime.Now + " " + swConv);
            swConv.Stop();    //컨베이어 생성 위치에 따라 Reset이 필요할 수 있음. --> Reset사용이 바람직.
            Thread.Sleep(1500);
            UnldCyl.Off();

            bInspExist = false;

            inspStep = AutoStep.step120;
          }
          break;

        case AutoStep.step120: //unldcyl 후진 확인하여 컴플릿               
          if (cmd == EqpCmd.ContiCycleStop)
          {
            inspStep = AutoStep.Complete;
          }
          else if (cmd == EqpCmd.CycleStop || cmd == EqpCmd.OneCycleStart)
          {
            OneCycle_InspStepEnd = true;
            inspStep = AutoStep.Ready;
          }
          break;


        // ======================== 적재 로직 ======================

        case AutoStep.step200: // 양품 conv on
          Convy.On();
          inspStep = AutoStep.step210;
          break;
        case AutoStep.step210: //ENDSEN 감지되면 CONV OFF
          if (io.X[(int)IOMapX.ConveyorEndSen] == true)           //io.X[16]
          {
            Convy.Off();
            Thread.Sleep(500);
            bUnldExist = true;
            bInspExist = false;
            inspStep = AutoStep.step220;
          }
          break;
        case AutoStep.step220: //CMD 확인하여 CYCLESTOP, ONE CYCLE START면 Ready Step으로 (STATUS = READY로 변경)
                               //CONTICYCLESTART면, STEP00으로(재공과 투입된 WORK가 없으면 STATUS=IDLE로)
          if (cmd == EqpCmd.ContiCycleStop)
          {
            inspStep = AutoStep.Complete;
          }
          else if (cmd == EqpCmd.CycleStop || cmd == EqpCmd.OneCycleStart)
          {
            OneCycle_InspStepEnd = true;
            inspStep = AutoStep.Ready;

          }
          break;
        case AutoStep.Complete:
          inspStep = AutoStep.Ready;


          break;
      }
    }

    void UnldSeq()   //검사 단계 sequence : 완성도 85%
    {
      switch (unldStep)
      {
        case AutoStep.Ready:
          if (io.X[(int)IOMapX.ConveyorEndSen] == true && OneCycle_UnldStepEnd == false)
          {
            unldStep = AutoStep.step00;
          }


          break;
        case AutoStep.step00:
          if (io.X[(int)IOMapX.ConveyorEndSen] == true)
          {
            MultMoveAbs(TeachList.Pos_LD);
            unldStep = AutoStep.step10;

          }
          else
          {
            unldStep = AutoStep.Ready;
          }
          break;
        case AutoStep.step10:
          if (XAxis.Info[AxisMap.MotionComplete] && YAxis.Info[AxisMap.MotionComplete])
          {
            PickCyl.On();
            Thread.Sleep(500);

            unldStep = AutoStep.step20;
          }
          break;
        case AutoStep.step20: //ENDSEN 감지 해제되면 컴플릿 
          if (PickCyl.IsOn())
          {
            PickVac.On();
            Thread.Sleep(500);

            unldStep = AutoStep.step30;
          }
          break;
        case AutoStep.step30: //ENDSEN 감지 해제되면 컴플릿 
          if (PickVac.IsOn())
          {
            PickCyl.Off();
            Thread.Sleep(500);
            bUnldExist = false;
            unldStep = AutoStep.step40;
          }
          break;
        case AutoStep.step40:
          if (PickCyl.IsOff())
          {
            var unloadPositions = new[]
            {
            TeachList.Pos1_1,
            TeachList.Pos1_2,
            TeachList.Pos1_3
          };

            if (nextUnloadIndex >= unloadPositions.Length)
            {
              OnMsg("사용 가능한 적재 위치가 없습니다.");
              sequenceFault = true;
              break;
            }

            string unloadPositionName = "Pos1_" + (nextUnloadIndex + 1);

            MultMoveAbs(unloadPositions[nextUnloadIndex]);
            Thread.Sleep(500);

            unldStep = AutoStep.step50;

            ProductUnloadPositionSelected?.Invoke(
                productInputTime,
                productInspectionResult,
                unloadPositionName);
          }
          break;
        case AutoStep.step50: //ENDSEN 감지 해제되면 컴플릿 
          if (XAxis.Info[AxisMap.MotionComplete] && YAxis.Info[AxisMap.MotionComplete])
          {
            PickCyl.On();
            Thread.Sleep(500);

            unldStep = AutoStep.step60;
          }
          break;
        case AutoStep.step60: //ENDSEN 감지 해제되면 컴플릿 
          if (PickCyl.IsOn())
          {
            PickVac.Off();
            Thread.Sleep(500);

            unldStep = AutoStep.step70;
          }
          break;
        case AutoStep.step70: //ENDSEN 감지 해제되면 컴플릿 
          if (PickVac.IsOff())
          {
            PickCyl.Off();
            Thread.Sleep(500);

            unldStep = AutoStep.step80;
          }
          break;
        case AutoStep.step80: //ENDSEN 감지 해제되면 컴플릿 
          if (PickCyl.IsOff())
          {
            MultMoveAbs(TeachList.Pos_LD);
            Thread.Sleep(500);
            nextUnloadIndex++;

            unldStep = AutoStep.step90;
          }
          break;
        case AutoStep.step90: //ENDSEN 감지 해제되면 컴플릿 
          if (XAxis.Info[AxisMap.MotionComplete] && YAxis.Info[AxisMap.MotionComplete])
          {

            Thread.Sleep(500);
            unldStep = AutoStep.step100;
          }
          break;
        case AutoStep.step100: //CMD 확인하여 CYCLESTOP, ONE CYCLE START면 Ready Step으로 (STATUS = READY로 변경)
                               //CONTICYCLESTART면, STEP00으로(재공과 투입된 WORK가 없으면 STATUS=IDLE로)
          if (cmd == EqpCmd.ContiCycleStop)
          {
            unldStep = AutoStep.Complete;
          }
          else if (cmd == EqpCmd.CycleStop || cmd == EqpCmd.OneCycleStart)
          {
            OneCycle_UnldStepEnd = true;
            unldStep = AutoStep.Ready;
          }
          break;
        case AutoStep.Complete:
          unldStep = AutoStep.Ready;


          break;

      }
    }

    void OnMsg(string msg)
    {
      if (OnMsgEvent != null) OnMsgEvent(msg);
    }

    public bool MultMoveAbs(TeachList pos)
    {
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
      if (ms != Status.MC_OK)
      {
        OnMsg($"{pos} MultMoveAbs Error : {ms}");
        return false;
      }
      OnMsg($"{pos} MultMoveAbs Start");
      return true;
    }

    public bool Home()
    {
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
      while (true)
      {
        Thread.Sleep(100);
        if (PickVac.IsOff() && PickCyl.IsOff()
            && XAxis.Info[AxisMap.IsHomed] && YAxis.Info[AxisMap.IsHomed])
          break;
        if (timeout++ > Global.HOME_TIMEOUT)
        {
          OnMsg($"장비 초기화 TimeOut : {Global.HOME_TIMEOUT / 10} 초");
          return false;
        }
      }
      OnMsg("장비 초기화 완료");
      return true;
    }

    public bool WorkMove(TeachList from, TeachList to)
    {
      int autoStep = 0, oldStep = -1, timeOut = 0;
      // 초기화 완료 여부
      if (!(PickVac.IsOff() && PickCyl.IsOff()
          && XAxis.Info[AxisMap.IsHomed] && YAxis.Info[AxisMap.IsHomed]))
      {
        OnMsg("장비 초기화 상태를 확인 하세요");
        return false;
      }

      while (true)
      {
        Thread.Sleep(100);

        // step간 100이상 정체시 timeout
        if (autoStep != oldStep)
        {
          timeOut = 0;
          oldStep = autoStep;
        }
        else
        {
          if (timeOut++ > 100)
          {
            OnMsg($"{autoStep} Step Time Out Error : {timeOut}");
            return false;
          }
        }

        switch (autoStep)
        {
          case 0:
            MultMoveAbs(from);
            Thread.Sleep(200);
            autoStep++;
            break;
          case 1:
            // from 이동 완료 체크
            //xAxis.Info[AxisMap.MotionComplete] 이동 완료
            if (XAxis.Info[AxisMap.MotionComplete] && YAxis.Info[AxisMap.MotionComplete])
            {
              // 실린더 하강
              PickCyl.On();
              autoStep++;
            }
            break;
          case 2:
            // 실린더 하강 확인
            if (PickCyl.IsOn())
            {
              // vac on
              PickVac.On();
              autoStep++;
            }
            break;
          case 3:
            // vac on 확인
            if (PickVac.IsOn())
            {
              // 실린더 상승
              PickCyl.Off();
              autoStep++;
            }
            break;
          case 4:
            // 실린더 상승 확인
            if (PickCyl.IsOff())
            {
              // to로 이동
              MultMoveAbs(to);
              Thread.Sleep(200);
              autoStep++;
            }
            break;
          case 5:
            // to 이동 확인
            if (XAxis.Info[AxisMap.MotionComplete] && YAxis.Info[AxisMap.MotionComplete])
            {
              // 실린더 하강
              PickCyl.On();
              autoStep++;
            }
            break;
          case 6:
            // 실린더 하강확인
            if (PickCyl.IsOn())
            {
              // vac off
              PickVac.Off();
              autoStep++;
            }
            break;
          case 7:
            // vac off 확인
            if (PickVac.IsOff())
            {
              // 시린더 상승
              PickCyl.Off();
              autoStep++;
            }
            break;
          case 8:
            if (PickCyl.IsOff())
            {
              // 시린더 상승 확인
              return true;
            }
            break;
        }
      }
    }






  }
}
