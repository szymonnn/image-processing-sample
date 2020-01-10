package org.pytorch.helloworld;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {

    TensorImage tensorImageBuffer = null;
    TensorBuffer tensorOutputBuffer = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Bitmap bitmap = null;
        Interpreter interpreter = null;
        GpuDelegate delegate = new GpuDelegate();
        try {
            // creating bitmap from packaged into app android asset 'image.jpg',
            // app/src/main/assets/image.jpg
            bitmap = BitmapFactory.decodeStream(getAssets().open("amber.jpg"));
            // loading serialized torchscript module from packaged into app android asset model.pt,
            // app/src/model/assets/model.pt
            Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
            interpreter = new Interpreter(assetFilePath(this, "converted_model.tflite"), options);
        } catch (IOException e) {
            Log.e("PytorchHelloWorld", "Error reading assets", e);
            finish();
        }

        long timeBeforeOperation = System.currentTimeMillis();
        int[] inputPixels = new int[bitmap.getHeight() * bitmap.getWidth()];
        bitmap.getPixels(inputPixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        Log.i("Time get pixels: ", "" + (System.currentTimeMillis() - timeBeforeOperation));
        float[] inputData = new float[bitmap.getHeight() * bitmap.getWidth() * 3];
        final int offset_g = inputPixels.length;
        final int offset_b = 2 * inputPixels.length;
        for (int i = 0; i < inputPixels.length; i++) {
            float r = Color.red(inputPixels[i]);
            float g = Color.green(inputPixels[i]);
            float b = Color.blue(inputPixels[i]);
            inputData[i] = r;
            inputData[offset_g + i] = g;
            inputData[offset_b + i] = b;
        }

        Log.i("Time normalization: ", "" + (System.currentTimeMillis() - timeBeforeOperation));

        tensorImageBuffer = new TensorImage(interpreter.getInputTensor(0).dataType());
        tensorOutputBuffer = TensorBuffer.createFixedSize(interpreter.getOutputTensor(0).shape(), interpreter.getOutputTensor(0).dataType());
        tensorImageBuffer.load(inputData, new int[]{224, 224, 3});

        Log.i("Time before inference: ", "" + (System.currentTimeMillis() - timeBeforeOperation));
        interpreter.run(tensorImageBuffer.getBuffer(), tensorOutputBuffer.getBuffer().rewind());

        Log.i("Time after inference: ", "" + (System.currentTimeMillis() - timeBeforeOperation));
        TensorProcessor tensorProcessor = new TensorProcessor.Builder().build();
        float[] output = tensorProcessor.process(tensorOutputBuffer).getFloatArray();

        // showing image on UI
        ImageView imageView = findViewById(R.id.image);

        Log.i("Time before bitmap: ", "" + (System.currentTimeMillis() - timeBeforeOperation));
        Bitmap newBitmap = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), Bitmap.Config.ARGB_8888);

        BitmapManipulator.setPixels(newBitmap, floatArrayToInt(output), newBitmap.getWidth(), newBitmap.getHeight());
        Log.i("Time after operations: ", "" + (System.currentTimeMillis() - timeBeforeOperation));
        imageView.setImageBitmap(newBitmap);
        delegate.close();
    }

    private int [] floatArrayToInt(float [] array) {
        int [] intArray = new int[array.length];
        for (int i = 0; i < array.length; i++) {
            intArray[i] = (int) array[i];
        }
        return intArray;
    }

    /**
     * Copies specified asset to the file in /files app directory and returns this file absolute path.
     *
     * @return absolute file path
     */
    public static File assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file;
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
            return file;
        }
    }
}
