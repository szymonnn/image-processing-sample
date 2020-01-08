package org.pytorch.helloworld;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Build;

import androidx.annotation.RequiresApi;

public class BitmapManipulator {
    public static void pixelsToIntArray(Bitmap bitmap, int[] data) {
        int pixelsCount = bitmap.getWidth() * bitmap.getHeight();
        int[] pixels = new int[pixelsCount];
        bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    }

    @RequiresApi(api = Build.VERSION_CODES.O)
    public static void setPixels(Bitmap bitmap, int[] data, int width, int height) {
        int[] pixels = new int[data.length/3];
        for (int i = 0; i < data.length/3; i++) {
            pixels[i] = Color.rgb(
                    data[i],
                    data[i + pixels.length],
                    data[i + 2 * pixels.length]
            );
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height);
    }
}
