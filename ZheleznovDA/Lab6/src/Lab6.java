import java.util.ArrayList;
import java.util.Collections;

public class Lab6 {

    public static void main(String[] args) {
        double a = 1;
        int N = 100;
        double l = 1;
        double h = l / N;
        double sigma = 0.25;
        double tau = Math.sqrt(sigma * h * h / a);

        int flag = 0;
        int tem = 1;

        double[][] u = explicitMethod(N, h, sigma, tau, flag, tem);

        int s = 25;

        double max = 0;
        double delta = 0;
        double temp;
        for (int i = 0; i < N; i++) {
            temp = Math.abs(u[s][i] - resultFunc(h * i, tau * s));
            delta += temp * temp;
            if(temp > max){
                max = temp;
            }
        }
        System.out.println("Явный метод 2Т1П");
        //System.out.println(Math.sqrt(delta) + " " + max);
        System.out.println(max);

        flag = 1;
        u = explicitMethod(N, h, sigma, tau, flag, tem);

        max = 0;
        delta = 0;

        for (int i = 0; i < N; i++) {
            temp = Math.abs(u[s][i] - resultFunc(h * i, tau * s));
            delta += temp * temp;
            if(temp > max){
                max = temp;
            }
        }
        System.out.println("\nЯвный метод 3Т2П");
        //System.out.println(Math.sqrt(delta) + " " + max);
        System.out.println(max);

        flag = 2;
        u = explicitMethod(N, h, sigma, tau, flag, tem);

        max = 0;
        delta = 0;

        for (int i = 0; i < N; i++) {
            temp = Math.abs(u[s][i] - resultFunc(h * i, tau * s));
            delta += temp * temp;
            if(temp > max){
                max = temp;
            }
        }
        System.out.println("\nЯвный метод 2Т2П");
        //System.out.println(Math.sqrt(delta) + " " + max);
        System.out.println(max);

        flag = 0;
        u = implicitMethod(N, h, sigma, tau, flag, tem);

        max = 0;
        delta = 0;

        for (int i = 0; i < N; i++) {
            temp = Math.abs(u[s][i] - resultFunc(h * i, tau * s));
            delta += temp * temp;
            if(temp > max){
                max = temp;
            }
        }
        System.out.println("\nНеявный метод 2Т1П");
        //System.out.println(Math.sqrt(delta) + " " + max);
        System.out.println(max);

        flag = 1;
        u = implicitMethod(N, h, sigma, tau, flag, tem);

        max = 0;
        delta = 0;

        for (int i = 0; i < N; i++) {
            temp = Math.abs(u[s][i] - resultFunc(h * i, tau * s));
            delta += temp * temp;
            if(temp > max){
                max = temp;
            }
        }
        System.out.println("\nНеявный метод 3Т2П");
        //System.out.println(Math.sqrt(delta) + " " + max);
        System.out.println(max);

        flag = 2;
        u = implicitMethod(N, h, sigma, tau, flag, tem);

        max = 0;
        delta = 0;

        for (int i = 0; i < N; i++) {
            temp = Math.abs(u[s][i] - resultFunc(h * i, tau * s));
            delta += temp * temp;
            if(temp > max){
                max = temp;
            }
        }
        System.out.println("\nНеявный метод 2Т2П");
        //System.out.println(Math.sqrt(delta) + " " + max);
        System.out.println(max);
    }

    static double resultFunc(double x, double t){
        return Math.exp(2*x) * Math.cos(t);
    }

    static double[][] explicitMethod(int N, double h, double sigma, double tau, int flag, int tem) {
        int K = N * 2;
        double[][] u = new double[K + 1][N + 1];

        //u[0][j]
        for (int i = 0; i < u[1].length; i++) {
            u[0][i] = Math.exp(2 * i * h);
        }

        //u[1][j]
        if(tem == 0) {
            for (int i = 0; i < u[1].length; i++) {
                u[1][i] = Math.exp(2 * i * h);
            }
        }
        else {
            for (int i = 0; i < u[1].length; i++) {
                u[1][i] = Math.exp(2 * i * h) + (4 * Math.exp(2 * i * h) - 5 * u[0][i]) * tau * tau / 2;
            }
        }

        for (int k = 1; k < K; k++) {
            for(int j = 1; j < N; j++){
                u[k + 1][j] = 2 * u[k][j] - u[k - 1][j] + sigma * (u[k][j + 1] - 2*u[k][j] + u[k][j - 1]) - 5 * tau * tau * u[k][j];
            }
            if(flag == 0) {
                u[k + 1][0] = u[k + 1][1] / (2 * h + 1);
                u[k + 1][N] = u[k + 1][N - 1] / (1 - 2 * h);
            }
            else if(flag == 1){
                u[k + 1][0] = (4 * u[k + 1][1] - u[k + 1][2]) / (3 + 4 * h) ;
                u[k + 1][N] = (4 * u[k + 1][N - 1] - u[k + 1][N - 2]) / (3 - 4 * h);
            }
            else if(flag == 2){
                u[k + 1][0] = (-(2 * u[k][0] - u[k - 1][0]) * h * h - 2 * tau * tau * u[k + 1][1]) / (-h * h - 2 * tau * tau - 5 * h * h * tau * tau - 4 * h * tau * tau);
                u[k + 1][N] = ((2 * u[k][N] - u[k - 1][N]) * h * h + 2 * tau * tau * u[k + 1][N - 1]) / (h * h + 2 * tau * tau + 5 * h * h * tau * tau - 4 * h * tau * tau);
            }
        }
        return u;
    }

    static double[][] implicitMethod(int N, double h, double sigma, double tau, int flag, int tem){
        int K = N * 2;
        double[][] u = new double[K + 1][N + 1];

        double[] a_arr = new double[N + 1];
        double[] b_arr = new double[N + 1];
        double[] c_arr = new double[N + 1];
        double[] d_arr = new double[N + 1];

        //u[0][j]
        for (int i = 0; i < u[1].length; i++) {
            u[0][i] = Math.exp(2 * i * h);
        }

        //u[1][j]
        if(tem == 0) {
            for (int i = 0; i < u[1].length; i++) {
                u[1][i] = Math.exp(2 * i * h);
            }
        }
        else {
            for (int i = 0; i < u[1].length; i++) {
                u[1][i] = Math.exp(2 * i * h) + (4 * Math.exp(2 * i * h) - 5 * u[0][i]) * tau * tau / 2;
            }
        }

        for (int i = 0; i < N + 1; i++) {
            if(i == 0){
                if(flag == 0){
                    a_arr[0] = 0;
                    b_arr[0] =  1 + 2 * h;
                    c_arr[0] = -1;
                }
                else if(flag == 1) {
                    a_arr[0] = 0;
                    b_arr[0] = 2 + 4 * h;
                    c_arr[0] = -4 + (1 + 2 * sigma + 5 * tau * tau) / sigma;
                }
                else if(flag == 2){
                    a_arr[0] = 0;
                    b_arr[0] = -(h * h + 2 * tau * tau + 5 * h * h * tau * tau + 4 * h * tau * tau);
                    c_arr[0] = 2 * tau * tau;
                }
            }
            else if(i < N) {
                a_arr[i] = sigma;
                b_arr[i] = -(1 + 2 * sigma + 5 * tau * tau);
                c_arr[i] = sigma;
            }
            else{
                if(flag == 0){
                    a_arr[i] = -1;
                    b_arr[i] = 1 - 2 * h;
                    c_arr[i] = 0;
                }
                else if(flag == 1){
                    a_arr[i] = -4 + (1 + 2 * sigma + 5 * tau * tau) / sigma;
                    b_arr[i] = 2 - 4 * h;
                    c_arr[i] = 0;
                }
                else if(flag == 2){
                    a_arr[i] = -2 * tau * tau;
                    b_arr[i] = h * h + 2 * tau * tau + 5 * h * h * tau * tau - 4 * h * tau * tau;
                    c_arr[i] = 0;
                }
            }
        }

        for(int k = 1; k < K; k++){
            for(int j = 0; j < N; j++){
                if(j == 0){
                    if(flag == 0) {
                        d_arr[0] = 0;
                    }
                    else if(flag == 1){
                        d_arr[0] = 2 * u[k][1] / sigma - u[k - 1][1] / sigma;
                    }
                    else if(flag == 2){
                        d_arr[0] = -h * h * (2 * u[k][0] - u[k - 1][0]);
                    }
                }
                else{
                    d_arr[j] = -2 * u[k][j] + u[k - 1][j];
                }
            }
            if(flag == 0) {
                d_arr[N] = 0;
            }
            else if(flag == 1){
                d_arr[N] = 2 / sigma * u[k][N - 1] - 1 / sigma * u[k - 1][N - 1];
            }
            else if(flag == 2){
                d_arr[N] = (2 * u[k][N] - u[k - 1][N]) * h * h;
            }

            ArrayList<Double> res_progonka = Progonka(a_arr, b_arr, c_arr, d_arr);
            for (int i = 0; i < N + 1; i++) {
                u[k + 1][i] = res_progonka.get(i);
            }
        }

        return u;
    }

    static ArrayList<Double> Progonka(double[] a, double[] b, double[] c, double[] d)
    {
        ArrayList<Double> roots = new ArrayList<>();
        ArrayList<Double> P = new ArrayList<>();
        ArrayList<Double> Q = new ArrayList<>();

        P.add(-c[0] / b[0]);
        Q.add(d[0] / b[0]);

        //Прямой ход
        for(int i = 1; i < a.length; i++)
        {
            P.add(-c[i] / (b[i] + a[i] * P.get(i - 1)));
            Q.add((d[i] - a[i] * Q.get(i - 1)) / (b[i] + a[i] * P.get(i - 1)));
        }

        Collections.reverse(P);
        Collections.reverse(Q);

        //Обратный ход
        roots.add(Q.get(0));
        for (int i = 1; i < a.length; i++)
        {
            roots.add(P.get(i) * roots.get(i - 1) + Q.get(i));
        }

        Collections.reverse(roots);
        return roots;
    }
}
