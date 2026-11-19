using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Helper {
  public enum IOMapX {
    LoadCylBwd = 0,   // X00 공급 실린더 후진
    LoadCylFwd = 1,   // X01 공급 실린더 전진
    WorkCylUp = 2,    // X02 가공 실린더 상승
    WorkCylDn = 3,    // X03 가공 실린더 하강
    MoveCylBwd = 4,   // X04 이송 실린더 후진
    MoveCylFwd = 5,   // X05 이송 실린더 전진
    UnldCyleBwd = 6,  // X06 취출 실린더 후진
    UnldCyleFwd = 7,  // X07 취출 실린더 전진
    PickCylUp = 8,    // X08 흡착 실린더 상승
    PickCylDn = 9,    // X09 흡착 실린더 하강
    PickVacSen1 = 10,  // X0A 흡착 센서1
    PickVacSen2 = 11,  // X0B 흡착 센서2
    MagazineSen = 12,  // X0C 매거진 검출
    InductiveSen = 13, // X0D 유도형
    CapacitiveSen = 14,// X0E 정전용량형
    PhotoSen = 15,     // X0F 광센서
    ConveyorEndSen = 16 // X10 컨베이어 종단
  }

  // 기존 보드 출력 비트 기준: Y20 -> 0, Y25 -> 5.
  public enum IOMapY {
    LoadCylFwd = 0,   // Y20 공급 실린더 전진 SOL
    LoadCylBwd = 1,   // Y21 공급 실린더 후진 SOL
    WorkCylDn = 2,    // Y22 가공 실린더 하강 SOL (편솔)
    MoveCylFwd = 3,   // Y23 이송 실린더 전진 SOL (편솔)
    UnldCyleFwd = 4,  // Y24 취출 실린더 전진 SOL (편솔)
    PickCylDn = 5,    // Y25 흡착 실린더 하강 SOL (편솔)
    VacOn = 6,        // Y26 흡착 ON (PickCyl용)
    VacOff = 7,       // Y27 흡착 OFF (PickCyl용)
    Drill = 8,        // Y28 가동 드릴
    Conveyor = 9,     // Y29 컨베이어 
    LampRed = 10,     // Y2A 경광등 적
    LampYellow = 11,  // Y2B 경광등 황
    LampGreen = 12    // Y2C 경광등 녹
  }
}
