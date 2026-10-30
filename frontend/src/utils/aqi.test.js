import { describe, expect, it } from "vitest";
import {
  BREAKPOINTS,
  getAQIInfo,
  healthIcon,
  pm25Color,
  pm25ToEpaAqi,
  pm25ToGaugePct,
} from "./aqi.js";

describe("pm25ToEpaAqi (EPA May 2024 breakpoints)", () => {
  it.each([
    [0, 0],
    [9.0, 50],
    [9.1, 51],
    [35.4, 100],
    [35.5, 101],
    [55.4, 150],
    [55.5, 151],
    [125.4, 200],
    [125.5, 201],
    [225.4, 300],
    [225.5, 301],
    [325.4, 500],
  ])("pm25 %f -> AQI %i", (pm25, aqi) => {
    expect(pm25ToEpaAqi(pm25)).toBe(aqi);
  });

  it("interpolates linearly inside a band", () => {
    expect(pm25ToEpaAqi(4.5)).toBe(25);
    expect(pm25ToEpaAqi(20)).toBe(71);
  });

  it("truncates to 0.1 rather than rounding, as EPA specifies", () => {
    expect(pm25ToEpaAqi(9.09)).toBe(50); // 9.09 -> 9.0, still good
    expect(pm25ToEpaAqi(35.49)).toBe(100); // 35.49 -> 35.4, not 35.5
  });

  it("clamps negative concentrations to 0 and values off the scale to 500", () => {
    expect(pm25ToEpaAqi(-3)).toBe(0);
    expect(pm25ToEpaAqi(1000)).toBe(500);
  });

  it("accepts numeric strings", () => {
    expect(pm25ToEpaAqi("35.5")).toBe(101);
  });

  it("is a non-decreasing integer in [0, 500] across the whole scale", () => {
    let prev = 0;
    for (let i = 0; i <= 4000; i++) {
      const aqi = pm25ToEpaAqi(i / 10);
      expect(Number.isInteger(aqi)).toBe(true);
      expect(aqi).toBeGreaterThanOrEqual(prev);
      expect(aqi).toBeLessThanOrEqual(500);
      prev = aqi;
    }
  });
});

describe("pm25Color", () => {
  const hex = /^#[0-9a-f]{6}$/i;

  it.each([
    [0, "#90ee90"], // Good, light end
    [9, "#00b894"], // Good, upper edge
    [13, "#ffd700"], // Moderate, upper edge
    [17, "#e8590c"], // Elevated, upper edge
    [55, "#800000"], // High saturates at 55
    [500, "#800000"],
  ])("pm25 %f -> %s", (pm25, color) => {
    expect(pm25Color(pm25).toLowerCase()).toBe(color);
  });

  it("restarts at each band's light end just past the boundary", () => {
    expect(pm25Color(9.0001).toLowerCase()).toBe("#ffff99");
    expect(pm25Color(13.0001).toLowerCase()).toBe("#ffb347");
    expect(pm25Color(17.0001).toLowerCase()).toBe("#ff6b6b");
  });

  it("returns the lightest green for negative input", () => {
    expect(pm25Color(-5).toLowerCase()).toBe("#90ee90");
  });

  it("always returns a 6-digit hex color", () => {
    for (let i = 0; i <= 800; i++) {
      expect(pm25Color(i / 10)).toMatch(hex);
    }
  });
});

describe("getAQIInfo", () => {
  // Band edges are inclusive on the upper side (<=), similar to in the backedn
  it.each([
    [0, "Good"],
    [9.0, "Good"],
    [9.01, "Moderate"],
    [13.0, "Moderate"],
    [13.01, "Elevated"],
    [17.0, "Elevated"],
    [17.01, "High"],
    [250, "High"],
  ])("pm25 %f -> %s", (pm25, category) => {
    const info = getAQIInfo(pm25);
    expect(info.category).toBe(category);
    expect(info.aqi_range).toBe(category);
  });

  it("returns the same color as pm25Color and a complete info object", () => {
    for (const pm25 of [2, 9.5, 15, 40]) {
      const info = getAQIInfo(pm25);
      expect(info.color).toBe(pm25Color(pm25));
      expect(info.bg).toMatch(/^rgba\(/);
      expect(info.label).toContain("µg/m³");
      expect(info.health_msg.length).toBeGreaterThan(0);
    }
  });
});

describe("BREAKPOINTS (legend)", () => {
  it("lists the four categories in order with an open-ended last band", () => {
    expect(BREAKPOINTS.map((b) => b.category)).toEqual(["Good", "Moderate", "Elevated", "High"]);
    expect(BREAKPOINTS.at(-1).max).toBe(Infinity);
  });

  it("agrees with getAQIInfo at each finite upper edge", () => {
    for (const { max, category } of BREAKPOINTS.slice(0, -1)) {
      expect(getAQIInfo(max).category).toBe(category);
    }
  });

  it("has a legend swatch equal to the map color at the band edge", () => {
    const [good, moderate, elevated, high] = BREAKPOINTS;
    expect(good.color).toBe(pm25Color(9));
    expect(moderate.color).toBe(pm25Color(13));
    expect(elevated.color).toBe(pm25Color(17));
    expect(high.color).toBe(pm25Color(35));
  });
});

describe("pm25ToGaugePct", () => {
  it("maps 0-20 µg/m³ onto 0-100 and caps at 100", () => {
    expect(pm25ToGaugePct(0)).toBe(0);
    expect(pm25ToGaugePct(10)).toBe(50);
    expect(pm25ToGaugePct(20)).toBe(100);
    expect(pm25ToGaugePct(80)).toBe(100);
  });
});

describe("healthIcon", () => {
  it("returns a distinct icon per category and a fallback for unknown ones", () => {
    const icons = ["Good", "Moderate", "Elevated", "High"].map(healthIcon);
    expect(new Set(icons).size).toBe(4);
    expect(healthIcon("Unknown")).toBe("?");
    expect(healthIcon(undefined)).toBe("?");
  });
});
