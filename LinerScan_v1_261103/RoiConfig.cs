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
        // 설정 저장과 검사 시작 시 읽기가 겹쳐 불완전한 JSON을 읽지 않도록 보호합니다.
        internal static readonly object FileSync = new object();

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
