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
      AddLoadPositionButton();
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
    private void AddLoadPositionButton() {
      int buttonWidth = (btnTeach.Width - 6) / 2;
      btnTeach.Width = buttonWidth;
      btnTeach.Text = "ULD Teaching";

      var btnLoadPosition = new Button {
        Text = "LD 좌표 입력",
        Bounds = new Rectangle(btnTeach.Right + 6, btnTeach.Top, buttonWidth, btnTeach.Height),
        UseVisualStyleBackColor = true
      };
      btnLoadPosition.Click += EditLoadPosition;
      btnTeach.Parent.Controls.Add(btnLoadPosition);
    }

    private void EditLoadPosition(object sender, EventArgs e) {
      DPoint current;
      machine.Teach.TryGetLoadPosition(out current);

      using (var dialog = new Form {
        Text = "LD 좌표 입력",
        ClientSize = new Size(280, 120),
        FormBorderStyle = FormBorderStyle.FixedDialog,
        StartPosition = FormStartPosition.CenterParent,
        ShowInTaskbar = false,
        MinimizeBox = false,
        MaximizeBox = false
      }) {
        var xInput = new TextBox { Location = new Point(75, 12), Width = 190, Text = current?.X.ToString() ?? "" };
        var yInput = new TextBox { Location = new Point(75, 44), Width = 190, Text = current?.Y.ToString() ?? "" };
        var save = new Button { Text = "저장", Location = new Point(109, 80) };
        var cancel = new Button { Text = "취소", Location = new Point(190, 80), DialogResult = DialogResult.Cancel };

        save.Click += (o, args) => {
          double x, y;
          if (!double.TryParse(xInput.Text, out x) || !double.TryParse(yInput.Text, out y)
              || double.IsNaN(x) || double.IsNaN(y) || double.IsInfinity(x) || double.IsInfinity(y)) {
            MessageBox.Show(dialog, "X, Y 좌표를 숫자로 입력해 주세요.", "LD 좌표 입력");
            return;
          }

          machine.Teach[TeachList.LD] = new DPoint(x, y);
          try {
            machine.Teach.Save();
          }
          catch (System.IO.IOException ex) {
            machine.Teach.RestoreLoadPosition(current);
            MessageBox.Show(dialog, ex.Message, "LD 좌표 저장 실패");
            return;
          }
          catch (UnauthorizedAccessException ex) {
            machine.Teach.RestoreLoadPosition(current);
            MessageBox.Show(dialog, ex.Message, "LD 좌표 저장 실패");
            return;
          }

          dialog.DialogResult = DialogResult.OK;
        };

        dialog.Controls.AddRange(new Control[] {
          new Label { Text = "X (mm)", Location = new Point(12, 15), AutoSize = true },
          new Label { Text = "Y (mm)", Location = new Point(12, 47), AutoSize = true },
          xInput, yInput, save, cancel
        });
        dialog.AcceptButton = save;
        dialog.CancelButton = cancel;
        if (dialog.ShowDialog(this) == DialogResult.OK)
          TeachListViewUpdate();
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

      DPoint loadPosition;
      if (machine.Teach.TryGetLoadPosition(out loadPosition)) {
        var item = new ListViewItem(TeachList.LD.ToString());
        item.SubItems.Add(loadPosition.X.ToString());
        item.SubItems.Add(loadPosition.Y.ToString());
        lvTeach.Items.Add(item);
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
