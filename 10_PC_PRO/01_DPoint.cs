using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Helper {
  public class DPoint {
    public DPoint(double x, double y) {
      this.x = x;
      this.y = y;
    }
    double x, y;
    public double X { get { return x; } set { x = value; } }
    public double Y { get { return y; } set { y = value; } }

    public void SetValue(double x, double y) {
      this.x = x;
      this.y = y;
    }
    public override string ToString() {
      return $"{x.ToString()},{y.ToString()}";
    }
    public static bool TryParse(string str, out DPoint value) {
      //str = "100.0,200.0"
      try {
        var strs = str.Split(',');
        double x = double.Parse(strs[0]); //"100.0"
        double y = double.Parse(strs[1]); //"200.0" 
        value = new DPoint(x, y);
        return true;
      } catch (Exception) {
        value = new DPoint(0, 0);
        return false;
      }
    }
  }
}
