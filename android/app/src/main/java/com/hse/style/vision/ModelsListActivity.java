package com.hse.style.vision;

import android.content.Intent;
import android.os.Bundle;

import com.hse.style.AbstractListActivity;
import com.hse.style.ImageSendingActivity;
import com.hse.style.ListCardView;
import com.hse.style.R;

public class ModelsListActivity extends AbstractListActivity {

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    // TODO: change it to RecyclerView
    ListCardView first_item = findViewById(R.id.models_item_1);
    ListCardView second_item = findViewById(R.id.models_item_2);
    ListCardView third_item = findViewById(R.id.models_item_3);

    first_item.setOnClickListener(v -> {
      final Intent intent = new Intent(ModelsListActivity.this, ImageSendingActivity.class);
      intent.putExtra(Intent.EXTRA_REFERRER, "https://f88a78e0.ngrok.io/style/api/v1.0/pose");
      startActivity(intent);
    });
    second_item.setOnClickListener(v -> {
      final Intent intent = new Intent(ModelsListActivity.this, ImageSendingActivity.class);
      intent.putExtra(Intent.EXTRA_REFERRER, "https://f88a78e0.ngrok.io/style/api/v1.0/tshirt");
      intent.putExtra("grub", 0);
      startActivity(intent);
     });
    third_item.setOnClickListener(v -> {
      final Intent intent = new Intent(ModelsListActivity.this, ImageSendingActivity.class);
      intent.putExtra(Intent.EXTRA_REFERRER, "https://f88a78e0.ngrok.io/style/api/v1.0/tshirt");
      intent.putExtra("grub", 1);
      startActivity(intent);
    });
  }

  @Override
  protected int getListContentLayoutRes() {
    return R.layout.selection_models_list_content;
  }
}
