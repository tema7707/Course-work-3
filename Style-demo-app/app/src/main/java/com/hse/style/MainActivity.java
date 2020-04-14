package com.hse.style;

import android.content.Intent;
import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;

import com.hse.style.generation.GenerationListActivity;
import com.hse.style.vision.ModelsListActivity;

public class MainActivity extends AppCompatActivity {

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    findViewById(R.id.main_vision_click_view).setOnClickListener(v -> startActivity(new Intent(MainActivity.this, ModelsListActivity.class)));
    findViewById(R.id.main_generation_click_view).setOnClickListener(v -> startActivity(new Intent(MainActivity.this, GenerationListActivity.class)));
  }
}
