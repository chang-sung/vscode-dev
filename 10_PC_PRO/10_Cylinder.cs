using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Helper;

namespace MotionKit
{
    public class Cylinder : IActuator
    {
        readonly IO io;

        readonly IOMapY bitOn;
        readonly IOMapY? bitOff;

        readonly IOMapX isBitOn;
        readonly IOMapX isBitOff;

        // 편솔 실린더
        public Cylinder(IOMapY bitOn, IOMapX isBitOn, IOMapX isBitOff)
        {
            this.bitOn = bitOn;
            this.bitOff = null;
            this.isBitOn = isBitOn;
            this.isBitOff = isBitOff;

            io = IO.GetInstance();
        }

        // 복솔 실린더
        public Cylinder(IOMapY bitOn, IOMapY bitOff, IOMapX isBitOn, IOMapX isBitOff)
        {
            this.bitOn = bitOn;
            this.bitOff = bitOff;
            this.isBitOn = isBitOn;
            this.isBitOff = isBitOff;

            io = IO.GetInstance();
        }

        // 전진 또는 하강
        public void On()
        {
            if (bitOff.HasValue)
                io.Off((int)bitOff.Value);

            io.On((int)bitOn);
        }

        // 후진 또는 상승
        public void Off()
        {
            io.Off((int)bitOn);

            if (bitOff.HasValue)
                io.On((int)bitOff.Value);
        }

        public bool IsOn()
        {
            return io.X[(int)isBitOn]
                && !io.X[(int)isBitOff];
        }

        public bool IsOff()
        {
            return !io.X[(int)isBitOn]
                && io.X[(int)isBitOff];
        }
    }
}