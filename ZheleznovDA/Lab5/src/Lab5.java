import java.util.ArrayList;
import java.util.Collections;

public class Lab5 {
    public static void main(String[] args) {
        double a = 1;
        int N = 100;
        double l = Math.PI;
        double h = l / N;
        double sigma = 0.5;
        double tau = sigma * h * h / a;

        int flag = 0;

        double[][] u = explicitMethod(a, N, h, sigma, tau, flag);

        int s = 25;

        double max = 0;
        double delta = 0;
        double temp;
        for (int i = 0; i < N; i++) {
            temp = Math.abs(u[s][i] - resultFunc(a, h * i, tau * s));
            delta += temp * temp;
            if(temp > max){
                max = temp;
            }
        }
        System.out.println("Явный метод 2Т1П");
        //System.out.println(Math.sqrt(delta) + " " + max);
        System.out.println(max);

        flag = 1;
        u = explicitMethod(a, N, h, sigma, tau, flag);

        max = 0;
        delta = 0;

        for (int i = 0; i < N; i++) {
            temp = Math.abs(u[s][i] - resultFunc(a, h * i, tau * s));
            delta += temp * temp;
            if(temp > max){
                max = temp;
            }
        }
        System.out.println("\nЯвный метод 3Т2П");
        //System.out.println(Math.sqrt(delta) + " " + max);
        System.out.println(max);

        flag = 2;
        u = explicitMethod(a, N, h, sigma, tau, flag);

        max = 0;
        delta = 0;

        for (int i = 0; i < N; i++) {
            temp = Math.abs(u[s][i] - resultFunc(a, h * i, tau * s));
            delta += temp * temp;
            if(temp > max){
                max = temp;
            }
        }
        System.out.println("\nЯвный метод 2Т2П");
        //System.out.println(Math.sqrt(delta) + " " + max);
        System.out.println(max);

        flag = 0;
        u = implicitMethod(a, N, h, sigma, tau, flag);

        max = 0;
        delta = 0;

        for (int i = 0; i < N; i++) {
            temp = Math.abs(u[s][i] - resultFunc(a, h * i, tau * s));
            delta += temp * temp;
            if(temp > max){
                max = temp;
            }
        }
        System.out.println("\nНеявный метод 2Т1П");
        //System.out.println(Math.sqrt(delta) + " " + max);
        System.out.println(max);

        flag = 1;
        u = implicitMethod(a, N, h, sigma, tau, flag);

        max = 0;
        delta = 0;

        for (int i = 0; i < N; i++) {
            temp = Math.abs(u[s][i] - resultFunc(a, h * i, tau * s));
            delta += temp * temp;
            if(temp > max){
                max = temp;
            }
        }
        System.out.println("\nНеявный метод 3Т2П");
        //System.out.println(Math.sqrt(delta) + " " + max);
        System.out.println(max);

        flag = 2;
        u = implicitMethod(a, N, h, sigma, tau, flag);

        max = 0;
        delta = 0;

        for (int i = 0; i < N; i++) {
            temp = Math.abs(u[s][i] - resultFunc(a, h * i, tau * s));
            delta += temp * temp;
            if(temp > max){
                max = temp;
            }
        }
        System.out.println("\nНеявный метод 2Т2П");
        //System.out.println(Math.sqrt(delta) + " " + max);
        System.out.println(max);

        flag = 0;
        u = CrankNicolson(a, N, h, sigma, tau, flag);

        max = 0;
        delta = 0;

        for (int i = 0; i < N; i++) {
            temp = Math.abs(u[s][i] - resultFunc(a, h * i, tau * s));
            delta += temp * temp;
            if(temp > max){
                max = temp;
            }
        }
        System.out.println("\nМетод Кранка-Николсона 2Т1П");
        //System.out.println(Math.sqrt(delta) + " " + max);
        System.out.println(max);

        flag = 1;
        u = CrankNicolson(a, N, h, sigma, tau, flag);

        max = 0;
        delta = 0;

        for (int i = 0; i < N; i++) {
            temp = Math.abs(u[s][i] - resultFunc(a, h * i, tau * s));
            delta += temp * temp;
            if(temp > max){
                max = temp;
            }
        }
        System.out.println("\nМетод Кранка-Николсона 3Т2П");
        //System.out.println(Math.sqrt(delta) + " " + max);
        System.out.println(max);

        flag = 2;
        u = CrankNicolson(a, N, h, sigma, tau, flag);

        max = 0;
        delta = 0;

        for (int i = 0; i < N; i++) {
            temp = Math.abs(u[s][i] - resultFunc(a, h * i, tau * s));
            delta += temp * temp;
            if(temp > max){
                max = temp;
            }
        }
        System.out.println("\nМетод Кранка-Николсона 2Т2П");
        //System.out.println(Math.sqrt(delta) + " " + max);
        System.out.println(max);
    }

    static double resultFunc(double a, double x, double t){
        return Math.exp(-a * t) * Math.sin(x);
    }

    static double[][] explicitMethod(double a, int N, double h, double sigma, double tau, int flag) {
        int K = N * 2;
        double[][] u = new double[K + 1][N + 1];

        for (int i = 0; i < u[1].length; i++) {
            u[0][i] = Math.sin(i * h);
        }

        for (int k = 0; k < K; k++) {
            for(int j = 1; j < N; j++){
                u[k + 1][j] = sigma * u[k][j + 1] + (1 - 2 * sigma) * u[k][j] + sigma * u[k][j - 1];
            }
            if(flag == 0) {
                u[k + 1][0] = u[k + 1][1] - h * Math.exp(-a * (k + 1) * tau);
                u[k + 1][N] = u[k + 1][N - 1] - h * Math.exp(-a * (k + 1) * tau);
            }
            else if(flag == 1){
                u[k + 1][0] = (-2 * h * Math.exp(-a * (k + 1) * tau) + 4 * u[k + 1][1] - u[k + 1][2]) / 3;
                u[k + 1][N] = (-2 * h * Math.exp(-a * (k + 1) * tau) + 4 * u[k + 1][N - 1] - u[k + 1][N - 2]) / 3;
            }
            else if(flag == 2){
                u[k + 1][0] = (u[k + 1][1] + h * h / (2 * a * tau) * u[k][0] - h * Math.exp(-a * (k + 1) * tau)) / (1 + h * h / (2 * a * tau));
                u[k + 1][N] = (u[k + 1][N - 1] + h * h / (2 * a * tau) * u[k][N] - h * Math.exp(-a * (k + 1) * tau)) / (1 + h * h / (2 * a * tau));
            }
        }
        return u;
    }

    static double[][] implicitMethod(double a, int N, double h, double sigma, double tau, int flag){
        int K = N * 2;
        double[][] u = new double[K + 1][N + 1];

        //sigma = 0.5;

        double[] a_arr = new double[N + 1];
        double[] b_arr = new double[N + 1];
        double[] c_arr = new double[N + 1];
        double[] d_arr = new double[N + 1];

        for (int i = 0; i < u[1].length; i++) {
            u[0][i] = Math.sin(i * h);
        }

        double koef = 1 / (2 * h * sigma);

        for (int i = 0; i < N + 1; i++) {
            if(i == 0){
                if(flag == 0){
                    a_arr[0] = 0;
                    b_arr[0] = -1 / h;
                    c_arr[0] = 1 / h;
                }
                else if(flag == 1) {
                    a_arr[0] = 0;
                    b_arr[0] = -1 / h;
                    c_arr[0] = 1 / h - koef;
                }
                else if(flag == 2){
                    a_arr[0] = 0;
                    b_arr[0] = 2 * a / h + h / tau;
                    c_arr[0] = -2 * a / h;
                }
            }
            else if(i < N) {
                a_arr[i] = sigma;
                b_arr[i] = -(1 + 2 * sigma);
                c_arr[i] = sigma;
            }
            else{
                if(flag == 0){
                    a_arr[i] = -1 / h;
                    b_arr[i] = 1 / h;
                    c_arr[i] = 0;
                }
                else if(flag == 1){
                    a_arr[i] = -1 / h + koef;
                    b_arr[i] = 1 / h;
                    c_arr[i] = 0;
                }
                else if(flag == 2){
                    a_arr[i] = -2 * a / h;
                    b_arr[i] = 2 * a / h + h / tau;
                    c_arr[i] = 0;
                }
            }
        }

        for(int k = 0; k < K; k++){
            for(int j = 0; j < N; j++){
                if(j == 0){
                    if(flag == 0) {
                        d_arr[0] = Math.exp(-a * (k + 1) * tau);
                    }
                    else if(flag == 1){
                        d_arr[0] = Math.exp(-a * (k + 1) * tau) - koef * u[k][1];
                    }
                    else if(flag == 2){
                        d_arr[0] = h / tau * u[k][0] - 2 * a * Math.exp(-a * (k + 1) * tau);
                    }
                }
                else{
                    d_arr[j] = -u[k][j];
                }
            }
            if(flag == 0) {
                d_arr[N] = -Math.exp(-a * (k + 1) * tau);
            }
            else if(flag == 1){
                d_arr[N] = -Math.exp(-a * (k + 1) * tau) + koef * u[k][N - 1];
            }
            else if(flag == 2){
                d_arr[N] = h / tau * u[k][N] - 2 * a * Math.exp(-a * (k + 1) * tau);
            }

            ArrayList<Double> res_progonka = Progonka(a_arr, b_arr, c_arr, d_arr);
            for (int i = 0; i < N + 1; i++) {
                u[k + 1][i] = res_progonka.get(i);
            }
        }

        return u;
    }

    static double[][] CrankNicolson(double a, int N, double h, double sigma, double tau, int flag){
        int K = N * 2;
        double[][] u = new double[K + 1][N + 1];

        //sigma = 0.5;

        double[] a_arr = new double[N + 1];
        double[] b_arr = new double[N + 1];
        double[] c_arr = new double[N + 1];
        double[] d_arr = new double[N + 1];

        for (int i = 0; i < u[1].length; i++) {
            u[0][i] = Math.sin(i * h);
        }

        double koef = 1 / (2 * h * sigma);
        double r = a * tau / (h * h);

        for (int i = 0; i < N + 1; i++) {
            if(i == 0){
                if(flag == 0){
                    a_arr[0] = 0;
                    b_arr[0] = -1 / h;
                    c_arr[0] = 1 / h;
                }
                else if(flag == 1) {
                    a_arr[0] = 0;
                    b_arr[0] = -1 / h;
                    c_arr[0] = 1 / h - koef;
                }
                else if(flag == 2){
                    a_arr[0] = 0;
                    b_arr[0] = 2 * a / h + h / tau;
                    c_arr[0] = -2 * a / h;
                }
            }
            else if(i < N) {
                a_arr[i] = -r / 2;
                b_arr[i] = r + 1;
                c_arr[i] = -r / 2;
            }
            else{
                if(flag == 0){
                    a_arr[i] = -1 / h;
                    b_arr[i] = 1 / h;
                    c_arr[i] = 0;
                }
                else if(flag == 1){
                    a_arr[i] = -1 / h + koef;
                    b_arr[i] = 1 / h;
                    c_arr[i] = 0;
                }
                else if(flag == 2){
                    a_arr[i] = -2 * a / h;
                    b_arr[i] = 2 * a / h + h / tau;
                    c_arr[i] = 0;
                }
            }
        }

        for(int k = 0; k < K; k++){
            for(int j = 0; j < N; j++){
                if(j == 0){
                    if(flag == 0) {
                        d_arr[0] = Math.exp(-a * (k + 1) * tau);
                    }
                    else if(flag == 1){
                        d_arr[0] = Math.exp(-a * (k + 1) * tau) - koef * u[k][1];
                    }
                    else if(flag == 2){
                        d_arr[0] = h / tau * u[k][0] - 2 * a * Math.exp(-a * (k + 1) * tau);
                    }
                }
                else{
                    d_arr[j] = r / 2 * (u[k][j - 1] + u[k][j + 1]) + u[k][j] * (1 - r);
                }
            }
            if(flag == 0) {
                d_arr[N] = -Math.exp(-a * (k + 1) * tau);
            }
            else if(flag == 1){
                d_arr[N] = -Math.exp(-a * (k + 1) * tau) + koef * u[k][N - 1];
            }
            else if(flag == 2){
                d_arr[N] = h / tau * u[k][N] - 2 * a * Math.exp(-a * (k + 1) * tau);
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
