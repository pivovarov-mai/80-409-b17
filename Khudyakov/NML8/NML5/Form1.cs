using System;
using System.Windows.Forms;
using System.Collections.Generic;

namespace NML5
{
    public partial class Form1 : Form
    {
        
        static int M;
        static int T;
        static int K = 100;
        static int nuvar = 2;
        static int Nx = 20;
        static int Ny = Nx;
        double a = 1;
        double[,,] U = new double[Nx + 1, Ny + 1, K + 1];
        double[,] dt = new double[Nx + 1, K + 1];
        double[,,] U2 = new double[Nx + 1, Ny + 1, K + 1];
        double[,] dt2 = new double[Nx + 1, K + 1];
        static double[,] nu = { { 1, 1 }, { 2, 1 }, { 1, 2 } };
        static double nu1 = nu[nuvar, 0];
        static double nu2 = nu[nuvar, 1];
        static double lx = Math.PI/2 * nu1;
        static double ly = Math.PI/2 * nu2;
        double hx = lx / Nx;
        double hy = ly / Ny;
        double tau = 0.01;
        double Phi1(double y, double t)
        {
            return Math.Cos(nu2 * y) * Math.Exp(-(Math.Pow(nu1, 2) + Math.Pow(nu2, 2)) * this.a * t);
        }
        double Phi2(double y, double t)
        {
            return Math.Cos(nu1 * lx) * Math.Cos(nu2 * y) * Math.Exp(-(Math.Pow(nu1, 2) + Math.Pow(nu2, 2)) * this.a * t);
        }
        double Phi3(double x, double t)
        {
            return Math.Cos(nu1 * x) * Math.Exp(-(Math.Pow(nu1, 2) + Math.Pow(nu2, 2)) * this.a * t);
        }
        double Phi4(double x, double t)
        {
            return Math.Cos(nu1 * x) * Math.Cos(nu2 * ly) * Math.Exp(-(Math.Pow(nu1, 2) + Math.Pow(nu2, 2)) * this.a * t);
        }
        public Form1()
        {
            InitializeComponent();
            trackBar1.Minimum = 0;
            trackBar1.Maximum = Nx;
        }
        
        private void paintY()
        {
            double u, x ;
            double y = M * hy;
            chart1.Series[0].Points.Clear();
            chart1.ChartAreas[0].AxisY.LabelStyle.Format = "f2";
            chart1.ChartAreas[0].AxisX.LabelStyle.Format = "f2";
            chart1.ChartAreas[0].AxisX.Maximum = lx;
            chart1.ChartAreas[0].AxisX.Minimum = 0;
            chart1.ChartAreas[0].AxisY.Maximum = 1;
            chart1.ChartAreas[0].AxisY.Minimum = -1;
            for (double i = 0; i <= lx; i += hx)
            {
                x = i;
                u = Math.Cos(nu1 * x) * Math.Cos(nu2 * y) * Math.Exp(-(Math.Pow(nu1, 2) + Math.Pow(nu2, 2)) * a * T * tau);
                chart1.Series[0].Points.AddXY(x, u);
            }
        }
        private void paintX()
        {
            double u, y;
            double x = M * hx;
            chart1.Series[0].Points.Clear();
            chart1.ChartAreas[0].AxisY.LabelStyle.Format = "f2";
            chart1.ChartAreas[0].AxisX.LabelStyle.Format = "f2";
            chart1.ChartAreas[0].AxisX.Maximum = ly;
            chart1.ChartAreas[0].AxisX.Minimum = 0;
            chart1.ChartAreas[0].AxisY.Maximum = 1;
            chart1.ChartAreas[0].AxisY.Minimum = -1;
            for (double i = 0; i <= ly; i += hy)
            {
                y = i;
                u = Math.Cos(nu1 * x) * Math.Cos(nu2 * y) * Math.Exp(-(Math.Pow(nu1, 2) + Math.Pow(nu2, 2)) * a * T * tau);
                chart1.Series[0].Points.AddXY(y, u);
            }
        }
        private double[,,] MPN()
        {
            double[,,] u = new double[Nx + 1, Ny + 1, K + 1];
            double[] b = new double[Nx + 1];
            double[] a = new double[Nx];
            double[] c = new double[Nx];
            double[] d = new double[Nx + 1];
            double[,] u1 = new double[Nx + 1, Ny + 1];
            double[,] u2 = new double[Nx + 1, Ny + 1];
            double[] x;
            double sigx = this.a * tau / (2 * Math.Pow(hx, 2));
            double sigy = this.a * tau / (2 * Math.Pow(hy, 2));
            for (int i = 0; i <= Nx; i++)
                for (int j = 0; j <= Ny; j++) u[i, j, 0] = Math.Cos(nu1 * hx * i) * Math.Cos(nu2 * hy * j);
            for(int k = 1; k <= K; k++)
            {
                double t = k * tau - tau / 2;
                for (int j = 1; j <= Ny - 1; j++)
                {
                    for (int i = 0; i <= Nx - 1; i++)
                    {
                        a[i] = -sigx; b[i] = 1 + 2 * sigx; c[i] = -sigx; d[i] = sigy * (u[i, j + 1, k - 1] - 2 * u[i, j, k - 1] + u[i, j - 1, k - 1]) + u[i, j, k - 1];
                    }
                    b[0] = 1; c[0] = 0; d[0] = Phi1(hy * j, t);
                    a[Nx - 1] = 0; b[Nx] = 1; d[Nx] = Phi2(hy * j, t);
                    x = Progon(a, b, c, d).ToArray();
                    for (int i = 0; i <= Nx; i++)
                    {
                        u1[i, j] = x[i];
                        u1[i, 0] = Phi3(hx * i, t);
                        u1[i, Ny] = Phi4(hx * i, t);
                    }
                }
                for (int j = 0; j <= Ny; j++)
                {
                    u1[0, j] = Phi1(hy * j, t);
                    u1[Nx, j] = Phi2(hy * j, t);
                }
                for (int i = 1; i <= Nx - 1; i++)
                {
                    for (int j = 0; j <= Ny - 1; j++)
                    {
                        a[j] = -sigy; b[j] = 1 + 2 * sigy; c[j] = -sigy; d[j] = sigx * (u1[i + 1, j] - 2 * u1[i, j] + u1[i - 1, j]) + u1[i, j];
                    }
                    b[0] = 1; c[0] = 0; d[0] = Phi3(hx * i, k * tau);
                    a[Ny - 1] = 0; b[Ny] = 1; d[Ny] = Phi4(hx * i, k * tau);
                    x = Progon(a, b, c, d).ToArray();
                    for (int j = 0; j <= Ny; j++)
                    {
                        u2[i, j] = x[j];
                        u2[0, j] = Phi1(hy * j, k * tau);
                        u2[Nx, j] = Phi2(hy * j, k * tau);
                    }
                }
                for (int i = 0; i <= Nx; i++)
                {
                    u1[i, 0] = Phi3(hx * i, k * tau);
                    u1[i, Ny] = Phi4(hx * i, k * tau);
                }
                for (int i = 0; i <= Nx; i++)
                    for (int j = 0; j <= Ny; j++) u[i, j, k] = u2[i, j];
            }
            return u;
        }
        private double[,,] MDS()
        {
            double[,,] u = new double[Nx + 1, Ny + 1, K + 1];
            double[] b = new double[Nx + 1];
            double[] a = new double[Nx];
            double[] c = new double[Nx];
            double[] d = new double[Nx + 1];
            double[,] u1 = new double[Nx + 1, Ny + 1];
            double[,] u2 = new double[Nx + 1, Ny + 1];
            double[] x;
            double sigx = this.a * tau / Math.Pow(hx, 2);
            double sigy = this.a * tau / Math.Pow(hy, 2);
            for (int i = 0; i <= Nx; i++)
                for (int j = 0; j <= Ny; j++) u[i, j, 0] = Math.Cos(nu1 * hx * i) * Math.Cos(nu2 * hy * j);
            for (int k = 1; k <= K; k++)
            {
                double t = k * tau - tau / 2;
                for (int j = 1; j <= Ny - 1; j++)
                {
                    for (int i = 0; i <= Nx - 1; i++)
                    {
                        a[i] = -sigx; b[i] = 1 + 2 * sigx; c[i] = -sigx; d[i] = u[i, j, k - 1];
                    }
                    b[0] = 1; c[0] = 0; d[0] = Phi1(hy * j, t);
                    a[Nx - 1] = 0; b[Nx] = 1; d[Nx] = Phi2(hy * j, t);
                    x = Progon(a, b, c, d).ToArray();
                    for (int i = 0; i <= Nx; i++)
                    {
                        u1[i, j] = x[i];
                        u1[i, 0] = Phi3(hx * i, t);
                        u1[i, Ny] = Phi4(hx * i, t);
                    }
                }
                for (int j = 0; j <= Ny; j++)
                {
                    u1[0, j] = Phi1(hy * j, t);
                    u1[Nx, j] = Phi2(hy * j, t);
                }
                for (int i = 1; i <= Nx - 1; i++)
                {
                    for (int j = 0; j <= Ny - 1; j++)
                    {
                        a[j] = -sigy; b[j] = 1 + 2 * sigy; c[j] = -sigy; d[j] = u1[i, j];
                    }
                    b[0] = 1; c[0] = 0; d[0] = Phi3(hx * i, k * tau);
                    a[Ny - 1] = 0; b[Ny] = 1; d[Ny] = Phi4(hx * i, k * tau);
                    x = Progon(a, b, c, d).ToArray();
                    for (int j = 0; j <= Ny; j++)
                    {
                        u2[i, j] = x[j];
                        u2[0, j] = Phi1(hy * j, k * tau);
                        u2[Nx, j] = Phi2(hy * j, k * tau);
                    }
                }
                for (int i = 0; i <= Nx; i++)
                {
                    u1[i, 0] = Phi3(hx * i, k * tau);
                    u1[i, Ny] = Phi4(hx * i, k * tau);
                }
                for (int i = 0; i <= Nx; i++)
                    for (int j = 0; j <= Ny; j++) u[i, j, k] = u2[i, j];
            }
            return u;
        }

        private double f(double x, double y, double t) // Точное значение функции
        {
            return Math.Cos(nu[nuvar, 0] * x) * Math.Cos(nu[nuvar, 1] * y) * Math.Exp(-(Math.Pow(nu[nuvar, 0], 2) + Math.Pow(nu[nuvar, 1], 2)) * a * t);
        }
        static List<Double> Progon(double[] arr, double[] b, double[] crr, double[] d) // Метод прогонки
        {
            double[] a = new double[arr.Length + 1];
            double[] c = new double[arr.Length + 1];
            a[0] = 0; c[crr.Length] = 0;
            for (int i = 1; i < a.Length; i++) a[i] = arr[i - 1];
            for (int i = 0; i < a.Length - 1; i++) c[i] = crr[i];
            List<Double> roots = new List<double>();
            List<Double> P = new List<double>();
            List<Double> Q = new List<double>();
            P.Add(-c[0] / b[0]);
            Q.Add(d[0] / b[0]);
            for (int i = 1; i < a.Length; i++)
            {
                P.Add(-c[i] / (b[i] + a[i] * P[(i - 1)]));
                Q.Add((d[i] - a[i] * Q[(i - 1)]) / (b[i] + a[i] * P[(i - 1)]));
            }
            P.Reverse();
            Q.Reverse();
            roots.Add(Q[0]);
            for (int i = 1; i < a.Length; i++)
            {
                roots.Add(P[i] * roots[i - 1] + Q[i]);
            }
            roots.Reverse();
            return roots;
        }
        private void chart1_Paint(object sender, PaintEventArgs e)
        {
            chart1.Series[0].Points.Clear();
            chart1.Series[1].Points.Clear();
            chart1.Series[2].Points.Clear();
            textBox1.Text = (Math.Round(dt[M, T] / (Nx + 1), 15)).ToString();
            textBox2.Text = (Math.Round(dt2[M, T] / (Nx + 1), 15)).ToString();
            if (radioButton2.Checked)
            {
                for (int i = 0; i <= Nx; i++)
                {
                    chart1.Series[1].Points.AddXY(i * hx, U[i, M, T]);
                    chart1.Series[2].Points.AddXY(i * hx, U2[i, M, T]);
                }
                paintY();
            }
            else if (radioButton1.Checked)
            {

                for (int j = 0; j <= Ny; j++)
                {
                    chart1.Series[1].Points.AddXY(j * hy, U[M, j, T]);
                    chart1.Series[2].Points.AddXY(j * hy, U2[M, j, T]);
                }
                paintX();
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            U = MPN();
            U2 = MDS();
            for (int k = 0; k <= K; k++)
            {
                for (int i = 0; i <= Nx; i++)
                {
                    for (int j = 0; j <= Ny; j++)
                    {
                        dt[i, k] += Math.Abs(U[i, j, k] - f(hx * i, hy * j, k * tau));
                        dt2[i, k] += Math.Abs(U2[i, j, k] - f(hx * i, hy * j, k * tau));
                    }
                }
            }
        }

        private void trackBar1_Scroll(object sender, EventArgs e)
        {
            M = trackBar1.Value;
            textBox4.Text = (M * hx).ToString();
        }
        private void trackBar2_Scroll(object sender, EventArgs e)
        {
            T = trackBar2.Value;
            textBox5.Text = (T * tau).ToString();
        }

        private void radioButton2_CheckedChanged(object sender, EventArgs e)
        {
            trackBar1.Minimum = 0;
            trackBar1.Maximum = Ny;
            trackBar1.Value = 0;
            M = 0;
            textBox4.Text = (0).ToString();
            label2.Text = "Значение Y:";
        }

        private void radioButton1_CheckedChanged(object sender, EventArgs e)
        {
            trackBar1.Minimum = 0;
            trackBar1.Maximum = Nx;
            trackBar1.Value = 0;
            M = 0;
            textBox4.Text = (0).ToString();
            label2.Text = "Значение X:";
        }
    }
}
