package com.hse.style.generation;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;

import com.hse.style.AbstractListActivity;
import com.hse.style.ImageSendingActivity;
import com.hse.style.ListCardView;
import com.hse.style.R;

public class ClothesListActivity extends AbstractListActivity {

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    int index = getIntent().getIntExtra(Intent.EXTRA_INDEX, 0);

    ListCardView[] items = new ListCardView[] {
            findViewById(R.id.clothes_item_1),
            findViewById(R.id.clothes_item_2),
            findViewById(R.id.clothes_item_3),
            findViewById(R.id.clothes_item_4),
            findViewById(R.id.clothes_item_5),
            findViewById(R.id.clothes_item_6)
    };
    TextView title = findViewById(R.id.clothes_list_title);
    title.setText(index == 0 ? "T-shirts" : "Sweaters");

    for (int i = index; i < index + 3; i++) {
      items[i].setVisibility(View.VISIBLE);
      int finalI = i;
      items[i].setOnClickListener(v -> {
        final Intent intent = new Intent(ClothesListActivity.this, ImageSendingActivity.class);
        intent.putExtra(Intent.EXTRA_INDEX, finalI);
        startActivity(intent);
      });
    }
  }

  @Override
  protected int getListContentLayoutRes() {
    return R.layout.selection_clothes_list_content;
  }
}
