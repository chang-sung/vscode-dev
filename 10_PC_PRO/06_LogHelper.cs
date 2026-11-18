using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace Helper {
  public class LogHelper {
    string filePath;
    public LogHelper(string path) {
      filePath = path;
      Directory.CreateDirectory(filePath);
    }
    public void Write(string msg) {
      string fileName = DateTime.Now.ToString("yyyyMMdd") + ".log";
      string path = Path.Combine(filePath, fileName);

      //int num = 3;
      //myMethod(num);

      //using (FileStream fs = new FileStream(path, FileMode.Append)) {
      //  using (StreamWriter sw = new StreamWriter(fs)) {

      //  }
      //}

      using (StreamWriter sw = new StreamWriter
                (new FileStream(path, FileMode.Append))) {
        sw.WriteLine(DateTime.Now.ToString("HH:mm:ss") + ":" + msg);
      }

      //using var sw = new StreamWriter(new FileStream(path, FileMode.Append));
      //visual studio 2019......
    }
  }
}
