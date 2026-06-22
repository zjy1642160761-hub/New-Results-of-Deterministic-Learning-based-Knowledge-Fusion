function dx = rossler_sys(~, x, p1, p2, p3)
    dx = [
        -x(2) - x(3);
        x(1) + p1 * x(2);
        p2 + x(3) * (x(1) - p3)
    ];
end