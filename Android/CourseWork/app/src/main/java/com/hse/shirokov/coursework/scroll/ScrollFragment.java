package com.hse.shirokov.coursework.scroll;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import com.hse.shirokov.coursework.R;
import com.hse.shirokov.coursework.scroll.model.CardModel;
import com.hse.shirokov.coursework.scroll.presenter.CardAdapter;

import java.util.ArrayList;
import java.util.List;

import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.RecyclerView;

public class ScrollFragment extends Fragment {
    List<CardModel> cards = new ArrayList<>();

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_scroll, container, false);
    }

    @Override
    public void onViewCreated(View view, Bundle savedState) {
        setInitialData();
        RecyclerView recyclerView = view.findViewById(R.id.cardsScroll);
        CardAdapter adapter = new CardAdapter(getContext(), cards);
        recyclerView.setAdapter(adapter);
    }

    private void setInitialData(){
        cards.add(new CardModel(
                "Adidas Y-3 Collection",
                "YOHJI LOVE TUBULAR",
                "https://st.tsum.com/btrx/i/10/53/44/58//01_1526.jpg",
                "https://lerayonfrais.fr/img/m/203-manu_medium.jpg")
        );
        cards.add(new CardModel(
                "Adidas Y-3 Collection",
                "YOHJI LOVE TUBULAR",
                "https://st.tsum.com/btrx/i/10/53/44/58//01_1526.jpg",
                "https://lerayonfrais.fr/img/m/203-manu_medium.jpg")
        );
        cards.add(new CardModel(
                "Adidas Y-3 Collection",
                "YOHJI LOVE TUBULAR",
                "https://st.tsum.com/btrx/i/10/53/44/58//01_1526.jpg",
                "https://lerayonfrais.fr/img/m/203-manu_medium.jpg")
        );
        cards.add(new CardModel(
                "Adidas Y-3 Collection",
                "YOHJI LOVE TUBULAR",
                "https://st.tsum.com/btrx/i/10/53/44/58//01_1526.jpg",
                "https://lerayonfrais.fr/img/m/203-manu_medium.jpg")
        );
    }
}
