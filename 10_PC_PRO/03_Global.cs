using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Helper {
  public class Global {
    public const ushort BOARD_ID = 0;
    public const ushort ECAT_ADDR = 1;
    public const int MAX_IO = 32;


    public const string X_AXIS_NAME = "X축";
    public const string Y_AXIS_NAME = "Y축";
    public const ushort X_AXIS_NUM = 2;
    public const ushort Y_AXIS_NUM = 3;
    public const int HOME_TIMEOUT = 200; //1 = 0.1sec

    public const double TableOffset = 34000;

    public const string ParamPath = "..\\..\\..\\..\\Param\\";
    public const string TeachPath = "..\\..\\..\\..\\Teach\\";
    public const string LogPath = "..\\..\\..\\..\\Log\\";
  }
}
