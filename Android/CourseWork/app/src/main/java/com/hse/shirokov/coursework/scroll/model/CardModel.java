package com.hse.shirokov.coursework.scroll.model;

public class CardModel {
    private String title, subtitle;
    private String imageUrl, logoUrl;

    public CardModel(String title, String subtitle, String imageUrl, String logoUrl) {
        this.title = title;
        this.subtitle = subtitle;
        this.imageUrl = imageUrl;
        this.logoUrl = logoUrl;
    }

    public String getTitle() {
        return title;
    }

    public String getSubtitle() {
        return subtitle;
    }

    public String getImageUrl() {
        return imageUrl;
    }

    public String getLogoUrl() {
        return logoUrl;
    }
}
