package org.pytorch.helloworld;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Bitmap bitmap = null;
        Module module = null;
        try {
            // creating bitmap from packaged into app android asset 'image.jpg',
            // app/src/main/assets/image.jpg
            bitmap = BitmapFactory.decodeStream(getAssets().open("640x480.jpg"));
            // loading serialized torchscript module from packaged into app android asset model.pt,
            // app/src/model/assets/model.pt
            module = Module.load(assetFilePath(this, "45_2019_12_16_1449_traced_640_480.pt"));
        } catch (IOException e) {
            Log.e("PytorchHelloWorld", "Error reading assets", e);
            finish();
        }

        // showing image on UI
        ImageView imageView = findViewById(R.id.image);
        imageView.setImageBitmap(bitmap);

        long timeBeforeOperation = System.currentTimeMillis();
        int[] inputPixels = new int[bitmap.getHeight() * bitmap.getWidth()];
        bitmap.getPixels(inputPixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        Log.i("Time get pixels: ", "" + (System.currentTimeMillis() - timeBeforeOperation));
        float[] inputData = new float[bitmap.getHeight() * bitmap.getWidth() * 3];
        float[] normMeanRGB = new float[]{0.485f, 0.456f, 0.406f};
        float[] normStdRGB = new float[]{0.229f, 0.224f, 0.225f};
        final int offset_g = inputPixels.length;
        final int offset_b = 2 * inputPixels.length;
        for (int i = 0; i < inputPixels.length; i++) {
            float r = Color.red(inputPixels[i]) / 255f;
            float g = Color.green(inputPixels[i]) / 255f;
            float b = Color.blue(inputPixels[i]) / 255f;
//            float rF = (r - normMeanRGB[0]) / normStdRGB[0];
//            float gF = (g - normMeanRGB[1]) / normStdRGB[1];
//            float bF = (b - normMeanRGB[2]) / normStdRGB[2];
            inputData[i] = r;
            inputData[offset_g + i] = g;
            inputData[offset_b + i] = b;
        }
        Log.i("Time normalization: ", "" + (System.currentTimeMillis() - timeBeforeOperation));

        final Tensor inputTensor = Tensor.fromBlob(inputData, new long[]{1, 3, bitmap.getHeight(), bitmap.getWidth()});
        // running the model

        Log.i("Time before inference: ", "" + (System.currentTimeMillis() - timeBeforeOperation));
        final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

        Log.i("Time after inference: ", "" + (System.currentTimeMillis() - timeBeforeOperation));
        // getting tensor content as java array of floats
        final float[] scores = outputTensor.getDataAsFloatArray();
        final float[] output = new float[scores.length];

        for (int i = 0; i< scores.length; i++) {
            output[i] = Math.min(255f, scores[i]);
        }

        Log.i("Time before bitmap: ", "" + (System.currentTimeMillis() - timeBeforeOperation));
        Bitmap newBitmap = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), Bitmap.Config.ARGB_8888);

        BitmapManipulator.setPixels(newBitmap, floatArrayToIntArray(output), newBitmap.getWidth(), newBitmap.getHeight());
        Log.i("Time after operations: ", "" + (System.currentTimeMillis() - timeBeforeOperation));
        imageView.setImageBitmap(newBitmap);
    }

    private int[] floatArrayToIntArray(float[] floatArray) {
        int[] intArray = new int[floatArray.length];
        for (int i = 0; i < intArray.length; i++) {
            intArray[i] = (int) floatArray[i];
        }
        return intArray;
    }

    /**
     * Copies specified asset to the file in /files app directory and returns this file absolute path.
     *
     * @return absolute file path
     */
    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
}
