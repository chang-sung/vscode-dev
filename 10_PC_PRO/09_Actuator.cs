using Helper;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MotionKit {
  public class Actuator :IActuator {

    public Actuator(IOMapY bitOn) {
      this.bitOn = bitOn;
  
      io = IO.GetInstance();
    }
        IOMapY bitOn;
    IOMapX isBitOn;
    IO io;

    public bool IsOff() {
      return !io.X[(int)isBitOn];
    }

    public bool IsOn() {
      return io.X[(int)isBitOn];
    }

    public void Off() {
      io.Off((int)bitOn);
    }

    public void On() {
      io.On((int)bitOn);

    }


  }
}
