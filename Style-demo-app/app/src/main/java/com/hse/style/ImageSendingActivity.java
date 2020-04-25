package com.hse.style;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.hse.style.utils.ImageUtils;

import java.util.concurrent.TimeUnit;

import okhttp3.FormBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class ImageSendingActivity extends AppCompatActivity {

    final static private String TAG = "DetectronClassification";
    final static private int GALLERY_REQUEST_CODE = 20;

    final static private String HOST = "https://f88a78e0.ngrok.io/style/api/v1.0/segmentate";
    final static private String KEY_FILE = "file";
    final static private String KEY_TYPE = "cloth";
    final static private String KEY_GRUB = "grub_cut";

    private ProgressBar sendingPb;
    private ImageView classificationPhoto;

    private int clothModel = 0;
    private int grubCut= 0;
    private String mHost;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_detectron_classification);

        clothModel = getIntent().getIntExtra(Intent.EXTRA_INDEX, 0);
        mHost = getIntent().getStringExtra(Intent.EXTRA_REFERRER);
        mHost = mHost != null ? mHost : HOST;
        grubCut = getIntent().getIntExtra("grub", 0);
        classificationPhoto = findViewById(R.id.image_classification_photo);
        sendingPb = findViewById(R.id.sending_pb);

        findViewById(R.id.select_button).setOnClickListener(view -> {
            pickFromGallery();
        });
    }

    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == Activity.RESULT_OK) {
            if (requestCode == GALLERY_REQUEST_CODE) {
                findViewById(R.id.select_image_tv).setVisibility(View.GONE);
                classificationPhoto.setPadding(10, 10, 10, 10);
                Uri selectedImage = data.getData();
                classificationPhoto.setImageURI(selectedImage);
                Bitmap bitmap = ((BitmapDrawable)classificationPhoto.getDrawable()).getBitmap();
                new ResponseTask().execute(ImageUtils.convert(bitmap));
            }
        }
    }

    private void pickFromGallery() {
        Intent intent=new Intent(Intent.ACTION_PICK);
        intent.setType("image/*");
        String[] mimeTypes = {"image/jpeg", "image/png"};
        intent.putExtra(Intent.EXTRA_MIME_TYPES,mimeTypes);
        startActivityForResult(intent, GALLERY_REQUEST_CODE);
    }

    private void showProgress() {
        classificationPhoto.setVisibility(View.INVISIBLE);
        sendingPb.setVisibility(View.VISIBLE);
    }

    private void hideProgress() {
        sendingPb.setVisibility(View.INVISIBLE);
        classificationPhoto.setVisibility(View.VISIBLE);
    }

    private void showError() {
        classificationPhoto.setPadding(100, 100, 100, 100);
        classificationPhoto.setImageDrawable(getDrawable(R.drawable.ic_error));
        hideProgress();
    }


    private class ResponseTask extends AsyncTask<String, Void, String> {
        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            showProgress();
        }

        @Override
        protected String doInBackground(String... path) {
            if (path == null || path.length == 0)
                return null;
            return getHttpResponse(path[0]);
        }

        @Override
        protected void onPostExecute(String content) {
            if (content == null) {
                hideProgress();
                Toast.makeText(ImageSendingActivity.this, "Error!", Toast.LENGTH_SHORT).show();
                showError();
                return;
            }
            Bitmap bitmap = ImageUtils.convert(content);
            classificationPhoto.setImageBitmap(bitmap);
            hideProgress();
        }

        private String getHttpResponse(String image) {
            OkHttpClient httpClient = new OkHttpClient.Builder()
                    .connectTimeout(10, TimeUnit.SECONDS)
                    .writeTimeout(10, TimeUnit.SECONDS)
                    .readTimeout(30, TimeUnit.SECONDS)
                    .build();
            RequestBody formBody = new FormBody.Builder()
                    .add(KEY_FILE, image)
                    .add(KEY_TYPE, String.valueOf(clothModel))
                    .add(KEY_GRUB, String.valueOf(grubCut))
                    .build();
            Request request = new Request.Builder()
                    .url(mHost)
                    .post(formBody)
                    .build();
            try {
                Response response = httpClient.newCall(request).execute();
                Log.i("AAAA", ""+response.code());
                if (response.code() != 200) {
                    return null;
                }
                return response.body().string();
            } catch (Exception ignored) {
                Log.i("AAAA", ""+ignored.getMessage());
                return null;
            }
        }
    }
}
