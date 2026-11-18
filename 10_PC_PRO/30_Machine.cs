using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Helper;
using NMCMotionSDK;
using System.Threading;

namespace MotionKit
{
    using Status = NMCSDKLib.MC_STATUS;
    using NMC = NMCSDKLib;

    public class Machine
    {
        //명사: x축, y축, 픽업실린더, 배큠, 테이블
        public Machine()
        {
            XAxis = new Servo(Global.X_AXIS_NAME, Global.X_AXIS_NUM);
            YAxis = new Servo(Global.Y_AXIS_NAME, Global.Y_AXIS_NUM);
            Teach = new TeachData("MainTeach");
            // 공급실린더: 복솔
            LoadCyl = new Cylinder(IOMapY.LoadCylFwd, IOMapY.LoadCylBwd, IOMapX.LoadCylFwd, IOMapX.LoadCylBwd);
            PickCyl = new Cylinder(IOMapY.PickCylDn, IOMapX.PickCylDn, IOMapX.PickCylUp);
            PickVac = new Vacuum(IOMapY.VacOn, IOMapY.VacOff, IOMapX.PickVacSen2);


            // Lamp
            LampRed = new Lamp(IOMapY.LampRed);
            LampYellow = new Lamp(IOMapY.LampYellow);
            LampGreen = new Lamp(IOMapY.LampGreen);
        }

        public Servo XAxis { get; }
        public Servo YAxis { get; }
        public TeachData Teach { get; }
        public IActuator PickCyl { get; }
        public IActuator PickVac { get; }
        public IActuator LoadCyl { get; }

        public Lamp LampRed { get; }
        public Lamp LampYellow { get; }
        public Lamp LampGreen { get; }

        public event Action<string> OnMsgEvent;
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
