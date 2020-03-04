package com.hse.style.generation;

import android.content.Intent;
import android.os.Bundle;

import com.hse.style.AbstractListActivity;
import com.hse.style.R;
import com.hse.style.vision.ImageClassificationActivity;

public class GenerationListActivity extends AbstractListActivity {

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    // TODO: change it to RecyclerView
    findViewById(R.id.tshirt_generation_click_area).setOnClickListener(v -> {
      final Intent intent = new Intent(GenerationListActivity.this, ImageClassificationActivity.class);
      startActivity(intent);
    });
    findViewById(R.id.hoodie_generation_click_area).setOnClickListener(v -> {
      final Intent intent = new Intent(GenerationListActivity.this, ImageClassificationActivity.class);
      startActivity(intent);
     });
    findViewById(R.id.skirt_generation_click_area).setOnClickListener(v -> {
      final Intent intent = new Intent(GenerationListActivity.this, ImageClassificationActivity.class);
      startActivity(intent);
    });
  }

  @Override
  protected int getListContentLayoutRes() {
    return R.layout.generation_list_content;
  }
}
