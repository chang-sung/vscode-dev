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

    public bool TryGetLoadPosition(out DPoint position) {
      return teach.TryGetValue(TeachList.LD, out position);
    }

    public void RestoreLoadPosition(DPoint position) {
      if (position == null)
        teach.Remove(TeachList.LD);
      else
        teach[TeachList.LD] = position;
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
      string loadPosition;
      DPoint loadPoint;
      if (dic.TryGetValue(TeachList.LD.ToString(), out loadPosition)
          && DPoint.TryParse(loadPosition, out loadPoint)) {
        teach[TeachList.LD] = loadPoint;
      }
      else {
        teach.Remove(TeachList.LD);
      }
      Save();
    }
    public void Save() {
      fileHelper.FileSave(teach);
    }
  }
}
