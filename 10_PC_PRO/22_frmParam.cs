using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Helper;

namespace MotionKit {
  public partial class frmParam : Form {
    //명사: ServoParam X축, Y축
    //동사: CreateListView, DoubleClickEvent_X, DoubleClickEvent_Y

    public frmParam(ServoParam _xParam, ServoParam _yParam) {
      InitializeComponent();
      xParam = _xParam;
      yParam = _yParam;
      CreateListView();
    }

    ServoParam xParam;
    ServoParam yParam;


    void CreateListView() {
      //xParam, yParam 현재값 출력
      for (ParamList i = 0; i < ParamList.MAX; i++) {
        ListViewItem lvi = new ListViewItem(i.ToString());
        lvi.SubItems.Add(xParam[i].ToString());
        lvX.Items.Add(lvi);

        lvi = new ListViewItem(i.ToString());
        lvi.SubItems.Add(yParam[i].ToString());
        lvY.Items.Add(lvi);
      }
    }

    private void lvX_MouseDoubleClick(object sender, MouseEventArgs e) {
      //팝업띄우고, xParam 값 저장
      if(lvX.SelectedItems.Count == 1) {
        string axis = "X Axis Parameter";
        string item = lvX.SelectedItems[0].SubItems[0].Text;
        string value = lvX.SelectedItems[0].SubItems[1].Text;

        frmPopup f = new frmPopup(axis, item, value);
        if(f.ShowDialog() == DialogResult.OK) {
          lvX.SelectedItems[0].SubItems[1].Text = f.Value.ToString();
          ParamList idx = (ParamList)Enum.Parse(typeof(ParamList), item);
          xParam[idx] = f.Value;
          xParam.Save();
        }
      }
    }

    private void lvY_MouseDoubleClick(object sender, MouseEventArgs e) {
      //팝업띄우고, yParam 값 저장
      if (lvY.SelectedItems.Count == 1) {
        string axis = "Y Axis Parameter";
        string item = lvY.SelectedItems[0].SubItems[0].Text;
        string value = lvY.SelectedItems[0].SubItems[1].Text;

        frmPopup f = new frmPopup(axis, item, value);
        if (f.ShowDialog() == DialogResult.OK) {
          lvY.SelectedItems[0].SubItems[1].Text = f.Value.ToString();
          ParamList idx = (ParamList)Enum.Parse(typeof(ParamList), item);
          yParam[idx] = f.Value;
          yParam.Save();
        }
      }
    }
  }
}
