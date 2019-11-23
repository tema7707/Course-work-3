package com.hse.shirokov.coursework.scroll.presenter;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import com.bumptech.glide.Glide;
import com.hse.shirokov.coursework.R;
import com.hse.shirokov.coursework.scroll.model.CardModel;

import java.util.List;

import androidx.recyclerview.widget.RecyclerView;

public class CardAdapter extends RecyclerView.Adapter<CardAdapter.ViewHolder> {
    private LayoutInflater inflater;
    private List<CardModel> cards;

    public CardAdapter(Context context, List<CardModel> phones) {
        this.cards = phones;
        this.inflater = LayoutInflater.from(context);
    }

    @Override
    public CardAdapter.ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        View view = inflater.inflate(R.layout.card_item, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(CardAdapter.ViewHolder holder, int position) {
        CardModel card = cards.get(position);
        holder.titleView.setText(card.getTitle());
        holder.subtitleView.setText(card.getSubtitle());
        Glide.with(holder.imageView.getContext())
             .load(card.getImageUrl())
             .centerInside()
             .placeholder(R.drawable.ic_launcher_background)
             .into(holder.imageView);
        Glide.with(holder.logoView.getContext())
                .load(card.getLogoUrl())
                .centerInside()
                .placeholder(R.drawable.ic_launcher_background)
                .into(holder.logoView);
    }

    @Override
    public int getItemCount() {
        return cards.size();
    }

    class ViewHolder extends RecyclerView.ViewHolder {
        final TextView titleView, subtitleView;
        final ImageView imageView, logoView;

        ViewHolder(View view){
            super(view);
            imageView = view.findViewById(R.id.cardImage);
            logoView = view.findViewById(R.id.cardLogo);
            titleView = view.findViewById(R.id.cardTitle);
            subtitleView = view.findViewById(R.id.cardSubtitle);
        }
    }
}
