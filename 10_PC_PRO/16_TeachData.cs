using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Helper;

namespace MotionKit {
  public class TeachData {
    public TeachData(string _teachingName) {
      filePath = Global.TeachPath + _teachingName;
      teach = new Dictionary<TeachList, DPoint>();
      fileHelper = new FileHelper(filePath);
      Load();
    }
    string filePath;
    Dictionary<TeachList, DPoint> teach;
    FileHelper fileHelper;

    public DPoint this[TeachList idx] {
      get { return teach[idx]; }
      set { teach[idx] = value; }
    }

    public void Load() {
      Dictionary<string, string> dic = fileHelper.FileLoad();

      for (TeachList item = 0; item < TeachList.Max; item++) {
        string key = item.ToString();
        DPoint value;
        if (dic.ContainsKey(key)) {
          DPoint.TryParse(dic[key], out value);
        }
        else {
          value = new DPoint(0, 0);
        }
        if (teach.ContainsKey(item) == false)
          teach.Add(item, value);
      }
      Save();
    }
    public void Save() {
      fileHelper.FileSave(teach);
    }
  }
}
