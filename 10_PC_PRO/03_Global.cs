using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Helper {

  public enum EqpStatus {
    Ready, //초기화 완료 (Run, Idle로 갈 수 있음)
    Stop,  //장비가 멈춘 상태 (초기화 안된 상태)
    Error, //동작 중 에러 상태
    Run,   //가동 상태 (재공 유)
    Idle   //가동 상태 (재공 무)
  }

  public enum EqpMode { Auto, Manual }

  public enum Inspector { Blue, Non_Blue }

  public enum EqpCmd {
    None,
    EStop,
    Init,
    OneCycleStart,
    ContiCycleStop,
    CycleStop
  }

  public enum AutoStep {
    Ready,
    step00, step10, step20, step30, step40, step50, step60, step70, step80, step90, step100,
    step110, step120, step130, step140, step150, step160, step170, step180, step190, step200,
    step210, step220, step230, step240, step250, step260, step270, step280, step290,
    Complete
  }

  public enum ColorChannel
  {
    Blue,
    Green,
    Red
  }

  public enum ColorLevel
  {
    Low,
    Upper
  }

  public class Global {

    
    public const int DRILL_TIME = 2000;
    public const int INSPECT_TIME    = 700;
    public const int UNLD_TIME = 700;

    public const ushort BOARD_ID = 0;
    public const ushort ECAT_ADDR = 1;
    public const int MAX_IO = 32;

    public const string X_AXIS_NAME = "X축";
    public const string Y_AXIS_NAME = "Y축";
    public const ushort X_AXIS_NUM = 2;
    public const ushort Y_AXIS_NUM = 3;
    public const int HOME_TIMEOUT = 200; //1 = 0.1sec

    public const double TableOffset = 34000;

    public const string ROI_Config_NAME = "ROI_Config";
    public const string Color_Config_NAME = "Color_Config";

    public const string ParamPath = "..\\..\\..\\Param\\";
    public const string TeachPath = "..\\..\\..\\Teach\\";
    public const string LogPath = "..\\..\\..\\Log\\";
  }
}