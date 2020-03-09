package com.hse.style.generation;

import android.content.Intent;
import android.os.Bundle;

import com.hse.style.AbstractListActivity;
import com.hse.style.ListCardView;
import com.hse.style.R;
import com.hse.style.vision.ImageClassificationActivity;

public class ClothesListActivity extends AbstractListActivity {

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    // TODO: change it to RecyclerView
    ListCardView first_item = findViewById(R.id.clothes_item_1);
    ListCardView second_item = findViewById(R.id.clothes_item_2);
    ListCardView third_item = findViewById(R.id.clothes_item_3);

    int[] titles, descriptions, images;
    Intent startedIntent = getIntent();
    if (startedIntent != null) {
      String category = startedIntent.getStringExtra("category");
      switch (category) {
        case ("T-shirts"):
          titles = ClothesData.tshirts_titles;
          descriptions = ClothesData.tshirts_descriptions;
          images = ClothesData.tshirts_images;
          break;
        default:
          titles = ClothesData.skirts_titles;
          descriptions = ClothesData.skirts_descriptions;
          images = ClothesData.skirts_images;
      }
    }

    first_item.setOnClickListener(v -> {
      final Intent intent = new Intent(ClothesListActivity.this, ImageClassificationActivity.class);
      startActivity(intent);
    });
    second_item.setOnClickListener(v -> {
      final Intent intent = new Intent(ClothesListActivity.this, ImageClassificationActivity.class);
      startActivity(intent);
     });
    third_item.setOnClickListener(v -> {
      final Intent intent = new Intent(ClothesListActivity.this, ImageClassificationActivity.class);
      startActivity(intent);
    });
  }

  @Override
  protected int getListContentLayoutRes() {
    return R.layout.selection_models_list_content;
  }
}
