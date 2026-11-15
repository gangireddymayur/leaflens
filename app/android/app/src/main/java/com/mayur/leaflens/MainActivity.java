package com.mayur.leaflens;

import android.os.Build;
import android.os.Bundle;
import android.view.Display;
import android.view.WindowManager;

import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestHighestRefreshRate();
    }

    // Ask Android for the display's highest refresh rate at the current resolution (for example
    // 120 Hz), so scrolling and animations feel smooth. Phones without such a mode are unaffected.
    private void requestHighestRefreshRate() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M) return;
        Display display = getWindowManager().getDefaultDisplay();
        Display.Mode current = display.getMode();
        Display.Mode best = current;
        for (Display.Mode mode : display.getSupportedModes()) {
            boolean sameSize = mode.getPhysicalWidth() == current.getPhysicalWidth()
                    && mode.getPhysicalHeight() == current.getPhysicalHeight();
            if (sameSize && mode.getRefreshRate() > best.getRefreshRate()) best = mode;
        }
        WindowManager.LayoutParams params = getWindow().getAttributes();
        params.preferredDisplayModeId = best.getModeId();
        getWindow().setAttributes(params);
    }
}
