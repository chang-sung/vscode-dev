using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Helper {
  public interface IActuator {
    void On();
    void Off();
    bool IsOn();
    bool IsOff();
  }
}
