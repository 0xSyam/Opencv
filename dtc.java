import org.opencv.core.*; // Mengimpor fungsionalitas inti OpenCV
import org.opencv.dnn.*; // Mengimpor fungsionalitas jaringan saraf dalam OpenCV
import org.opencv.imgcodecs.Imgcodecs; // Mengimpor codec gambar OpenCV
import org.opencv.imgproc.Imgproc; // Mengimpor fungsionalitas pemrosesan gambar OpenCV
import org.opencv.utils.Converters; // Mengimpor fungsi utilitas OpenCV
import org.opencv.highgui.HighGui; // Mengimpor fungsionalitas GUI tingkat tinggi OpenCV
import java.nio.file.Files; // Mengimpor utilitas file NIO Java
import java.nio.file.Paths; // Mengimpor utilitas Paths NIO Java
import java.io.IOException; // Mengimpor penanganan pengecualian IO Java

import java.util.ArrayList; // Mengimpor kelas ArrayList
import java.util.List; // Mengimpor antarmuka List

public class dtc {
    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); } // Memuat pustaka native OpenCV

    public static void main(String[] args) {
        String bobotModel = "C:\\Users\\ASUS\\AndroidStudioProjects\\hello\\yolo\\yolov3.weights"; // Jalur ke file bobot YOLO
        String konfigurasiModel = "C:\\Users\\ASUS\\AndroidStudioProjects\\hello\\yolo\\yolov3.cfg"; // Jalur ke file konfigurasi YOLO
        String jalurFile = "C:\\Users\\ASUS\\AndroidStudioProjects\\hello\\yolo\\coco.names"; // Jalur ke file nama kelas
        String jalurGambar = "C:\\Users\\ASUS\\AndroidStudioProjects\\hello\\assets\\a.png"; // Jalur ke gambar input

        // Memuat model YOLO
        Net jaringan;
        try {
            jaringan = Dnn.readNetFromDarknet(konfigurasiModel, bobotModel); // Memuat model YOLO dari konfigurasi dan bobot
        } catch (Exception e) {
            System.err.println("Error: Tidak dapat memuat model YOLO. Periksa jalur file konfigurasi dan bobot.");
            e.printStackTrace();
            return;
        }
        
        List<String> namaLapisan = dapatkanNamaLapisanKeluaran(jaringan); // Mendapatkan nama-nama lapisan keluaran

        // Membaca gambar input
        Mat gambar = Imgcodecs.imread(jalurGambar); // Membaca gambar input

        // Check if image is loaded successfully
        if (gambar.empty()) {
            System.err.println("Error: Gambar tidak dapat dimuat. Periksa jalur file: " + jalurGambar);
            return;
        }

        // Prakondisi gambar
        Mat blob = Dnn.blobFromImage(gambar, 1 / 255.0, new Size(416, 416), new Scalar(0, 0, 0), true, false); // Mengubah gambar menjadi blob

        // Menetapkan input ke jaringan
        jaringan.setInput(blob); // Menetapkan blob sebagai input ke jaringan

        // Melakukan forward pass
        List<Mat> hasil = new ArrayList<>(); // Daftar untuk menyimpan hasil forward pass
        jaringan.forward(hasil, namaLapisan); // Melakukan forward pass dan menyimpan hasilnya

        // Pasca-pemrosesan deteksi
        float ambangKepercayaan = 0.5f; // Ambang batas kepercayaan
        List<Integer> idKelas = new ArrayList<>(); // Daftar untuk menyimpan ID kelas
        List<Float> kepercayaan = new ArrayList<>(); // Daftar untuk menyimpan kepercayaan
        List<Rect2d> kotak = new ArrayList<>(); // Daftar untuk menyimpan kotak pembatas

        for (Mat level : hasil) { // Iterasi melalui setiap level hasil
            for (int i = 0; i < level.rows(); i++) { // Iterasi melalui setiap baris di level
                Mat baris = level.row(i); // Mendapatkan baris
                Mat skor = baris.colRange(5, level.cols()); // Mendapatkan skor
                Core.MinMaxLocResult mm = Core.minMaxLoc(skor); // Mendapatkan nilai minimum dan maksimum serta lokasinya
                float kepercayaanMaks = (float) mm.maxVal; // Mendapatkan kepercayaan maksimum
                Point titikIdKelas = mm.maxLoc; // Mendapatkan lokasi kepercayaan maksimum

                if (kepercayaanMaks > ambangKepercayaan) { // Jika kepercayaan lebih besar dari ambang batas
                    int pusatX = (int) (baris.get(0, 0)[0] * gambar.cols()); // Menghitung koordinat X pusat
                    int pusatY = (int) (baris.get(0, 1)[0] * gambar.rows()); // Menghitung koordinat Y pusat
                    int lebar = (int) (baris.get(0, 2)[0] * gambar.cols()); // Menghitung lebar
                    int tinggi = (int) (baris.get(0, 3)[0] * gambar.rows()); // Menghitung tinggi
                    int kiri = pusatX - lebar / 2; // Menghitung koordinat kiri
                    int atas = pusatY - tinggi / 2; // Menghitung koordinat atas

                    idKelas.add((int) titikIdKelas.x); // Menambahkan ID kelas ke daftar
                    kepercayaan.add(kepercayaanMaks); // Menambahkan kepercayaan ke daftar
                    kotak.add(new Rect2d(kiri, atas, lebar, tinggi)); // Menambahkan kotak pembatas ke daftar
                }
            }
        }

        // Menerapkan non-maxima suppression
        MatOfRect2d arrayKotak = new MatOfRect2d(); // Membuat MatOfRect2d untuk menyimpan kotak pembatas
        arrayKotak.fromList(kotak); // Mengonversi daftar kotak pembatas ke MatOfRect2d
        MatOfFloat arrayKepercayaan = new MatOfFloat(Converters.vector_float_to_Mat(kepercayaan)); // Mengonversi daftar kepercayaan ke MatOfFloat
        MatOfInt indeks = new MatOfInt(); // Membuat MatOfInt untuk menyimpan indeks kotak yang dipilih
        Dnn.NMSBoxes(arrayKotak, arrayKepercayaan, ambangKepercayaan, 0.4f, indeks); // Menerapkan non-maxima suppression

        // Memuat nama kelas
        List<String> namaKelas = new ArrayList<>(); // Daftar untuk menyimpan nama kelas
        try {
            namaKelas = Files.readAllLines(Paths.get(jalurFile)); // Membaca nama kelas dari file
        } catch (IOException e) {
            System.err.println("Error: Tidak dapat memuat nama kelas. Periksa jalur file: " + jalurFile);
            e.printStackTrace();
            return;
        }

        // Menggambar kotak pembatas dan label
        int jumlahObjek = 0; // Inisialisasi jumlah objek
        for (int i = 0; i < indeks.rows(); i++) { // Iterasi melalui indeks yang dipilih
            int idx = (int) indeks.get(i, 0)[0]; // Mendapatkan indeks
            Rect2d kotakPembatas = kotak.get(idx); // Mendapatkan kotak pembatas
            Imgproc.rectangle(gambar, kotakPembatas.tl(), kotakPembatas.br(), new Scalar(0, 255, 0), 2); // Menggambar kotak pembatas
            String label = namaKelas.get(idKelas.get(idx)) + ": " + kepercayaan.get(idx); // Membuat label
            Point posisiLabel = new Point(kotakPembatas.tl().x, kotakPembatas.tl().y - 10); // Menyesuaikan posisi label
            Imgproc.putText(gambar, label, posisiLabel, Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 255, 0)); // Menggambar label
            System.out.println((i + 1) + ". Detected: " + label); // Mencetak label klasifikasi ke konsol dengan penomoran
            jumlahObjek++; // Menambah jumlah objek
        }

        // Menampilkan jumlah objek pada gambar
        String teksJumlahObjek = "Objects detected: " + jumlahObjek; // Membuat teks jumlah objek
        Imgproc.putText(gambar, teksJumlahObjek, new Point(10, gambar.rows() - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(0, 0, 255), 2); // Menggambar teks jumlah objek

        // Menampilkan jumlah objek ke konsol
        System.out.println("Total objects detected: " + jumlahObjek);

        // Menampilkan gambar output
        HighGui.imshow("YOLO Object Detection", gambar); // Menampilkan gambar dengan kotak pembatas dan label
        HighGui.waitKey(); // Menunggu penekanan tombol
    }

    private static List<String> dapatkanNamaLapisanKeluaran(Net jaringan) {
        List<String> nama = new ArrayList<>(); // Daftar untuk menyimpan nama lapisan keluaran
        List<Integer> lapisanKeluaran = jaringan.getUnconnectedOutLayers().toList(); // Mendapatkan indeks lapisan keluaran yang tidak terhubung
        List<String> namaLapisan = jaringan.getLayerNames(); // Mendapatkan nama semua lapisan

        for (Integer item : lapisanKeluaran) { // Iterasi melalui indeks lapisan keluaran yang tidak terhubung
            nama.add(namaLapisan.get(item - 1)); // Menambahkan nama lapisan keluaran ke daftar
        }
        return nama; // Mengembalikan daftar nama lapisan keluaran
    }
}