<?php
declare(strict_types=1);

require __DIR__ . '/MathCaptchaClient.php';

use App\Services\MathCaptchaClient;

// Kalitni KODGA YOZMANG — env yoki konfiguratsiyadan oling.
$client = new MathCaptchaClient(
    'https://api.qobilbek.dev/mathocr',
    getenv('MATHOCR_API_KEY') ?: '',
    5.0,
    2,
    0.85   // ilova o'z chegarasini xizmatnikidan qattiqroq qo'yishi mumkin
);

$path = $argv[1] ?? __DIR__ . '/../samples/a740762c-d57d-48b4-9b24-a0a586d6b383.webp';
$r    = $client->recognizeFile($path);

if ($r->valid) {
    printf(
        "ifoda: %s\nchap: %d\noperator: %s\no'ng: %d\nnatija: %s\nishonch: %.2f (%.1f ms)\n",
        $r->expression, $r->leftOperand, $r->operator, $r->rightOperand,
        (string)$r->result, $r->confidence, $r->elapsedMs
    );
    exit(0);
}

fwrite(STDERR, "tanib bo'lmadi: " . $r->error . ' (ishonch ' . number_format($r->confidence, 2) . ")\n");
exit(1);
