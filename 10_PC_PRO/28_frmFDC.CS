using log4net;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Xml.Linq;

namespace MotionKit {
    public partial class frmFDC : Form {
    
    TcpListener listener;
    TcpClient client;
    NetworkStream stream;
    StreamWriter writer;

    byte[] readBuf;
    List<FdcConfig> configlist;
    List<FdcSVID> svidlist;

    const int FDC_INTERVAL = 1000;
    static bool bRunFlag = false;
    public event Action<string> ServerStatusChanged;

    private void UpdateServerStatus(string status){

        btnListen.Text = status;
        btnListen.ForeColor =
            status == "Server Running" ? Color.Blue : Color.Red;
        ServerStatusChanged?.Invoke(status);
    }

    static readonly ILog log =
        LogManager.GetLogger(MethodBase.GetCurrentMethod().DeclaringType);
    
    public frmFDC(){
        InitializeComponent();
        
        readBuf = new byte[1024 * 5];
        configlist = new List<FdcConfig>();
        svidlist = new List<FdcSVID>();
    }

    private void frmServer_Load(object sender, EventArgs e) {
        configlist.Add(new FdcConfig(){
            EqpID = "ACADEMY01",
            ModID = "academy01_PC00",
            ToolID = "1", TRID = "1", LotID ="10",
            RecipeID = "1001"
        });

        svidlist.Add(new FdcSVID(){
            SVID_ID = "1000", SVID_Value = "1234",
            SVID_Name = "XAxisOverLoad"
        });

        svidlist.Add(new FdcSVID() {
            SVID_ID = "1001", SVID_Value = "2345",
            SVID_Name = "YAxisOverLoad"
        });

        gridConfig.DataSource = configlist;
        gridSvid.DataSource = svidlist;

        tbMessage.ScrollBars = ScrollBars.Both;
        tbMessage.WordWrap = false;
        tbMessage.ScrollToCaret();
    }

    private async void btnListen_Click(object sender, EventArgs e) {
        if (bRunFlag) {
            MessageBox.Show("Server Stop");
            StopServer();
        }
        else {
            MessageBox.Show("Server Start");
            await StartServerAsync();
        }
    }

    public async Task StartServerAsync() {

        if (bRunFlag)
            return;
        try {
            writer?.Close();
            stream?.Close();
            client?.Close();
            listener?.Stop();

            listener = new TcpListener(IPAddress.Any, 9000);
            listener.Start();

            bRunFlag = true;

            UpdateServerStatus("Server Running");

            while (bRunFlag) {
                TcpClient acceptedClient;

                try {
                    acceptedClient = await listener.AcceptTcpClientAsync();
                }
                catch (ObjectDisposedException) {
                    break;
                }
                catch (SocketException) {
                    if (!bRunFlag)
                        break;
                    throw;
                }

                client = acceptedClient;

                _ = HandleClient(acceptedClient);
                _ = SendReports(acceptedClient);
            }
        }

        catch (Exception ex) {
            tbMessage.AppendText($"*** Error : {ex.Message}\r\n");
            log.Info($"Error : {ex.Message}");
        }
        finally {
            bRunFlag = false;
            listener?.Stop();
            UpdateServerStatus("Server Stopped");
        }
    }

    public void StopServer() {
        bRunFlag = false;
        listener?.Stop();
    }

    private async Task HandleClient(TcpClient client) {

        stream = client.GetStream();
        writer = new StreamWriter(stream);

        string firstMsg = MakeFirstMsg();
        await Write(firstMsg);

        if (cbxFirst.Checked) tbMessage.AppendText(firstMsg);
        log.Info($"{firstMsg}");

        byte[] buffer = new byte[1024 * 5];
        int readLength;

        while ((readLength = await stream.ReadAsync(buffer, 0, buffer.Length)) > 0){
            string message = Encoding.UTF8.GetString(buffer, 0, readLength);

            tbMessage.AppendText(message);

            if (message.Contains("<Command>NONSECSCHECK</Command>")) {
                string aliveMsg = MakeAliveMsg();
                await Write(aliveMsg);

                if (cbxAlive.Checked) tbMessage.AppendText(aliveMsg);
                log.Info($"{aliveMsg}");
            }
        }
    }

    private async Task SendReports(TcpClient client){

        while (bRunFlag){
            await Task.Delay(FDC_INTERVAL);

            string svidMsg = MakeSVIDMsg();
            await Write(svidMsg);

            if (cbxReport.Checked) tbMessage.AppendText(svidMsg);
            log.Info($"{svidMsg}");
        }
    }

    private void btnSend_Click(object sender, EventArgs e){
        string text = textBox1.Text;
        byte[] messageBuffer = Encoding.UTF8.GetBytes($"Server : {text}");
        MessageBox.Show($"{messageBuffer}");
        stream.Write(messageBuffer, 0, messageBuffer.Length);
        log.Info($"Send to Server : {text}");
    }

    private void btnSend_KeyDown(object sender, KeyEventArgs e) {
        if (e.KeyCode == Keys.Enter) {
            btnSend_Click(sender, e);
        }

        tbMessage.Clear();
        tbMessage.Focus();
    }

    private void btnClear_Click(object sender, EventArgs e) {
        tbMessage.Clear();
    }

    public async Task Write(string msg) {
        try {
            await writer.WriteAsync(msg);
            writer.Flush();
        }
        catch (Exception e){
            Console.WriteLine(e.Message);
        }
    }

    string MakeFirstMsg() {
        XElement msg = new XElement(Commons.Msg);
        XElement cmd = new XElement(Commons.Cmd, Commons.CmdAliveRep);
        XElement dataLayer = new XElement(Commons.DataLayer);
        XElement eqpid = new XElement(Commons.EqpID, configlist[0].EqpID);
        XElement modid = new XElement(Commons.ModID, configlist[0].ModID);

    

        msg.Add(cmd);
        msg.Add(dataLayer);
        dataLayer.Add(eqpid);
        dataLayer.Add(modid);

        Console.WriteLine(msg);
        return msg.ToString(SaveOptions.DisableFormatting);
    }

    string MakeAliveMsg() {
        
        XElement msg = new XElement(Commons.Msg);
        XElement cmd = new XElement(Commons.Cmd, Commons.CmdAliveRep);
        XElement dataLayer = new XElement(Commons.DataLayer);
        XElement time = new XElement(Commons.TimeStamp, DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"));
        
        XElement eqpid = new XElement(Commons.EqpID, configlist[0].EqpID);
        XElement modid = new XElement(Commons.ModID, configlist[0].ModID);
        XElement Toolid = new XElement(Commons.ToolID, configlist[0].ToolID);
        
        msg.Add(cmd);
        msg.Add(dataLayer);
        dataLayer.Add(time);
        dataLayer.Add(eqpid);
        dataLayer.Add(modid);
        dataLayer.Add(Toolid);

        Console.WriteLine(msg);
        return msg.ToString(SaveOptions.DisableFormatting);
    
    }

    public string MakeSVIDMsg() {

        string msg = Commons.CmdReport + " ";
        msg += "HDR =(null, null, null)";
        msg += $"TIME_STAMP = {DateTime.Now.ToString("yyy/MM/dd-HH:mm:ss.sss")} ";
        msg += $"EQPID = {configlist[0].EqpID}";
        msg += $"TRID = {configlist[0].TRID}";
        msg += $"LOTID = ({configlist[0].LotID})";
        msg += $"RECIPE = ({configlist[0].RecipeID})";
        msg += "SENSOR_VALUES = { ";

        int cnt = 1;
        foreach (var item in svidlist) {
            if (svidlist.Count == cnt++)
                msg += item.SVID_ID + "=" + item.SVID_Value + "}";
            else
                msg += item.SVID_ID + "=" + item.SVID_Value + "^";
        }
        Console.WriteLine(msg);
        return msg;
    }



    
    
    }
}