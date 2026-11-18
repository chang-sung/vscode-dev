using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Helper;

namespace MotionKit {
  public class Vacuum : IActuator {
    public Vacuum(IOMapY bitOn, IOMapY bitOff, IOMapX isBitOn) {
      this.bitOn = bitOn;
      this.bitOff = bitOff;
      this.isBitOn = isBitOn;
      io = IO.GetInstance();
    }
    IOMapY bitOn, bitOff;
    IOMapX isBitOn;
    IO io;
    public bool IsOff() {
      return !io.X[(int)isBitOn];
    }

    public bool IsOn() {
      return io.X[(int)isBitOn];
    }

    public void Off() {
      io.On((int)bitOff);
      io.Off((int)bitOn);
    }

    public void On() {
      io.On((int)bitOn);
      io.Off((int)bitOff);
    }
  }
}
