@echo off
echo ========================================
echo    Testing AQI Analysis Project
echo ========================================
echo.

echo [1/2] Running quick test...
".conda\python.exe" -c "
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
print('✓ All libraries imported successfully!')

# Test with small dataset
np.random.seed(42)
X = np.random.randn(100, 5)
y = np.random.randn(100)
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X, y)
score = model.score(X, y)
print(f'✓ Model training successful! R² score: {score:.3f}')

# Test plotting
plt.figure(figsize=(6, 4))
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title('Test Plot')
plt.savefig('test_plot.png', dpi=100, bbox_inches='tight')
plt.close()
print('✓ Plot generation successful!')
print('✓ All tests passed!')
"

echo [2/2] Quick test completed!
echo.

if exist "test_plot.png" (
    echo ✓ Test plot generated successfully
    del "test_plot.png"
) else (
    echo ✗ Test plot generation failed
)

echo.
echo Ready to run main analysis!
pause
