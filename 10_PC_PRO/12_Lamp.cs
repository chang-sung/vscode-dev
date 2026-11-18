using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Helper;

namespace MotionKit
{
    public class Lamp
    {
        readonly IO io;
        readonly IOMapY bit;

        public Lamp(IOMapY bit)
        {
            this.bit = bit;
            io = IO.GetInstance();
        }

        public void On()
        {
            io.On((int)bit);
        }

        public void Off()
        {
            io.Off((int)bit);
        }
    }
}