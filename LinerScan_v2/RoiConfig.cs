using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LinerScan.Imaging
{
    public class RoiConfig
    {
        public Rectangle Roi1_A { get; set; }
        public Rectangle Roi1_B { get; set; }
        public Rectangle Roi2_A { get; set; }
        public Rectangle Roi2_B { get; set; }

        public int Roi1_A_CropOffsetX { get; set; }
        public int Roi1_A_CropOffsetY { get; set; }
        public int Roi1_B_CropOffsetX { get; set; }
        public int Roi1_B_CropOffsetY { get; set; }

        public int Roi2_A_CropOffsetX { get; set; }
        public int Roi2_A_CropOffsetY { get; set; }
        public int Roi2_B_CropOffsetX { get; set; }
        public int Roi2_B_CropOffsetY { get; set; }
    }
}

