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
  public partial class frmTeach : Form {
    public frmTeach(Machine machine) {
      InitializeComponent();
      this.machine = machine;
      MoveRelativeBtnClickEventChain();
      TeachListViewUpdate();
    }

    Machine machine;

    void MoveRelativeBtnClickEventChain() {
      foreach (Control c in gbX.Controls) {
        if (c is Button)
          c.Click += btnMoveRelativeX;
      }
      foreach (Control c in gbY.Controls) {
        if (c is Button)
          c.Click += btnMoveRelativeY;
      }
    }
    void btnMoveRelativeX(object o, EventArgs e) {
      Button b = o as Button;
      if (b == null) return;

      double dist_mm;
      if (double.TryParse(b.Text, out dist_mm)) {
        // button text 가 숫자인 경우
        machine.XAxis.MoveRel(dist_mm);
      }
      else if (double.TryParse(tbDist.Text, out dist_mm)) {
        // button text move 인 경우
        machine.XAxis.MoveRel(dist_mm);
      }
    }

    void btnMoveRelativeY(object o, EventArgs e) {
      Button b = o as Button;
      if (b == null) return;

      double dist_mm;
      if (double.TryParse(b.Text, out dist_mm)) {
        // button text 가 숫자인 경우
        machine.YAxis.MoveRel(dist_mm);
      }
      else if (double.TryParse(tbDist.Text, out dist_mm)) {
        // button text move 인 경우
        machine.YAxis.MoveRel(dist_mm);
      }
    }
    void TeachListViewUpdate() {
      lvTeach.Items.Clear(); // 기존 데이터 제거

      for (TeachList list = 0; list < TeachList.Max; list++) {

        ListViewItem lvi = new ListViewItem(list.ToString());
        lvi.SubItems.Add(machine.Teach[list].X.ToString());
        lvi.SubItems.Add(machine.Teach[list].Y.ToString());

        lvTeach.Items.Add(lvi);
      }
    }

    private void btnJogYPos_MouseDown(object sender, MouseEventArgs e) {
      machine.YAxis.MoveVelPos();
    }
    private void btnJobXPos_MouseDown(object sender, MouseEventArgs e) {
      machine.XAxis.MoveVelPos();
    }
    private void btnJogYNeg_MouseDown(object sender, MouseEventArgs e) {
      machine.YAxis.MoveVelNeg();
    }
    private void btnJobXNeg_MouseDown(object sender, MouseEventArgs e) {
      machine.XAxis.MoveVelNeg();
    }
    void btnJogXStop(object o, MouseEventArgs e) {
      machine.XAxis.ServoHalt();
    }
    void btnJogYStop(object o, MouseEventArgs e) {
      machine.YAxis.ServoHalt();
    }

    private void btnTeach_Click(object sender, EventArgs e) {
      // 현재 x, y  좌표값
      double curXpos = machine.XAxis.Info.ActPos;
      double curYpos = machine.YAxis.Info.ActPos;
      // 21포인트에 대한 Teaching 

      for (TeachList pos = 0; pos < TeachList.Max; pos++) {
        int xIdx = (int)pos % 7;
        int yIdx = (int)pos / 7;

        double xpos = curXpos + (Global.TableOffset * xIdx);
        double ypos = curYpos + (Global.TableOffset * yIdx);

        machine.Teach[pos].SetValue(xpos, ypos);
      }

      // 저장
      machine.Teach.Save();
      TeachListViewUpdate();
    }

    private void btnMove_Click(object sender, EventArgs e) {
      if (lvTeach.SelectedItems.Count == 1) {
        string strPos = lvTeach.SelectedItems[0].SubItems[0].Text;

        TeachList pos = (TeachList)Enum.Parse(typeof(TeachList), strPos);
        machine.MultMoveAbs(pos);
      }
      else {
        MessageBox.Show("하나의 티칭 값을 선택해 주세요", "주의");
      }
    }

    private void btnCylUp_Click(object sender, EventArgs e) {
      machine.PickCyl.Off();
    }
    private void btnCylDn_Click(object sender, EventArgs e) {
      machine.PickCyl.On();
    }
    private void btnVacOn_Click(object sender, EventArgs e) {
      machine.PickVac.On();
    }
    private void btnVacOff_Click(object sender, EventArgs e) {
      machine.PickVac.Off();
    }



  }
}
