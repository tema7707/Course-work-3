package com.hse.style.generation;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Toast;

import com.hse.style.AbstractListActivity;
import com.hse.style.R;

public class GenerationListActivity extends AbstractListActivity {

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    // TODO: change it to RecyclerView
    findViewById(R.id.tshirt_generation_click_area).setOnClickListener(v -> {
      final Intent intent = new Intent(GenerationListActivity.this, ClothesListActivity.class);
      intent.putExtra(Intent.EXTRA_INDEX, 0);
      startActivity(intent);
    });
    findViewById(R.id.sweater_generation_click_area).setOnClickListener(v -> {
      final Intent intent = new Intent(GenerationListActivity.this, ClothesListActivity.class);
      intent.putExtra(Intent.EXTRA_INDEX, 3);
      startActivity(intent);
     });
    findViewById(R.id.skirt_generation_click_area).setOnClickListener(v -> {
      Toast.makeText(this, "Coming soon", Toast.LENGTH_SHORT).show();
    });
  }

  @Override
  protected int getListContentLayoutRes() {
    return R.layout.generation_list_content;
  }
}
