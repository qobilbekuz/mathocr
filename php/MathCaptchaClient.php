<?php
declare(strict_types=1);

namespace App\Services;

use CURLFile;
use finfo;
use RuntimeException;

/**
 * Python recognition xizmatiga mijoz.
 *
 * PHP 7.4 mos (serverdagi asosiy versiya), PHP 8.x da ham o'zgarishsiz ishlaydi.
 *
 * Ishlatish:
 *
 *     $client = new MathCaptchaClient('https://api.qobilbek.dev/mathocr');
 *     $r = $client->recognizeFile('/tmp/captcha.webp');
 *     if ($r->valid) {
 *         echo $r->expression . ' = ' . $r->result;
 *     } else {
 *         // MUHIM: taxmin qilmang. Rasmni qayta so'rang yoki qo'lda tekshiring.
 *         error_log('tanib bo\'lmadi: ' . $r->error);
 *     }
 */
final class RecognitionResult
{
    /** @var bool */
    public $valid;
    /** @var string|null  masalan "19+60" */
    public $expression;
    /** @var int|null */
    public $leftOperand;
    /** @var string|null  + - × ÷ */
    public $operator;
    /** @var int|null */
    public $rightOperand;
    /** @var int|float|null */
    public $result;
    /** @var float  0..1 */
    public $confidence;
    /** @var string|null */
    public $error;
    /** @var array<int,array<string,mixed>>  har bir belgi va uning bounding box'i */
    public $characters;
    /** @var float */
    public $elapsedMs;
    /** @var array<string,mixed>  xizmatdan kelgan to'liq javob */
    public $raw;

    /** @param array<string,mixed> $d */
    public static function fromArray(array $d): self
    {
        $r = new self();
        $r->valid        = (bool)($d['valid'] ?? false);
        $r->expression   = isset($d['expression']) ? (string)$d['expression'] : null;
        $r->leftOperand  = isset($d['left_operand']) ? (int)$d['left_operand'] : null;
        $r->operator     = isset($d['operator']) ? (string)$d['operator'] : null;
        $r->rightOperand = isset($d['right_operand']) ? (int)$d['right_operand'] : null;
        $r->result       = $d['result'] ?? null;
        $r->confidence   = (float)($d['confidence'] ?? 0.0);
        $r->error        = isset($d['error']) ? (string)$d['error'] : null;
        $r->characters   = is_array($d['characters'] ?? null) ? $d['characters'] : [];
        $r->elapsedMs    = (float)($d['elapsed_ms'] ?? 0.0);
        $r->raw          = $d;

        return $r;
    }
}

final class MathCaptchaClient
{
    private const MAX_BYTES    = 2097152;  // 2 MB
    private const ALLOWED_MIME = ['image/webp', 'image/png', 'image/jpeg', 'image/bmp'];

    /** @var string */
    private $baseUrl;
    /** @var float */
    private $timeout;
    /** @var int */
    private $retries;
    /** @var float */
    private $minConfidence;
    /** @var string  X-API-Key qiymati; bo'sh bo'lsa header yuborilmaydi */
    private $apiKey;

    public function __construct(
        string $baseUrl = 'https://api.qobilbek.dev/mathocr',
        string $apiKey = '',
        float $timeout = 5.0,
        int $retries = 2,
        float $minConfidence = 0.0
    ) {
        $this->baseUrl       = rtrim($baseUrl, '/');
        $this->apiKey        = $apiKey;
        $this->timeout       = $timeout;
        $this->retries       = max(0, $retries);
        $this->minConfidence = $minConfidence;
    }

    /** Diskdagi fayldan tanish. */
    public function recognizeFile(string $path): RecognitionResult
    {
        if (!is_file($path) || !is_readable($path)) {
            throw new RuntimeException("fayl o'qilmadi: {$path}");
        }
        $size = filesize($path);
        if ($size === false || $size === 0 || $size > self::MAX_BYTES) {
            throw new RuntimeException("fayl hajmi noto'g'ri: {$path}");
        }
        $mime = (new finfo(FILEINFO_MIME_TYPE))->file($path);
        if (!is_string($mime) || !in_array($mime, self::ALLOWED_MIME, true)) {
            throw new RuntimeException('qo\'llab-quvvatlanmaydigan format: ' . (string)$mime);
        }

        return $this->send('/recognize', ['image' => new CURLFile($path, $mime, basename($path))], false);
    }

    /** Xotiradagi baytlardan tanish (masalan yuklab olingan rasm). */
    public function recognizeBytes(string $bytes): RecognitionResult
    {
        $len = strlen($bytes);
        if ($len === 0 || $len > self::MAX_BYTES) {
            throw new RuntimeException('rasm hajmi noto\'g\'ri');
        }

        return $this->send('/recognize/base64', ['image_base64' => base64_encode($bytes)], true);
    }

    /**
     * Xizmat holati — deploy tekshiruvi va monitoring uchun.
     *
     * @return array<string,mixed>
     */
    public function health(): array
    {
        $ch = curl_init($this->baseUrl . '/health');
        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT        => (int)ceil($this->timeout),
            CURLOPT_CONNECTTIMEOUT => 2,
        ]);
        $body = curl_exec($ch);
        curl_close($ch);
        if (!is_string($body)) {
            return [];
        }
        $data = json_decode($body, true);

        return is_array($data) ? $data : [];
    }

    /**
     * @param array<string,mixed> $payload
     */
    private function send(string $path, array $payload, bool $asJson): RecognitionResult
    {
        $lastError = 'noma\'lum xato';

        for ($attempt = 0; $attempt <= $this->retries; $attempt++) {
            if ($attempt > 0) {
                usleep(100000 * (int)pow(2, $attempt - 1));  // 100ms, 200ms, ...
            }

            $ch      = curl_init($this->baseUrl . $path);
            $headers = $this->apiKey !== '' ? ['X-API-Key: ' . $this->apiKey] : [];
            $options = [
                CURLOPT_POST           => true,
                CURLOPT_RETURNTRANSFER => true,
                CURLOPT_TIMEOUT        => (int)ceil($this->timeout),
                CURLOPT_CONNECTTIMEOUT => 2,
                CURLOPT_FAILONERROR    => false,
            ];
            if ($asJson) {
                $json = json_encode($payload);
                if ($json === false) {
                    return RecognitionResult::fromArray(['valid' => false, 'error' => 'json_encode xatosi']);
                }
                $options[CURLOPT_POSTFIELDS] = $json;
                $headers[] = 'Content-Type: application/json';
            } else {
                $options[CURLOPT_POSTFIELDS] = $payload;  // multipart/form-data
            }
            if ($headers !== []) {
                $options[CURLOPT_HTTPHEADER] = $headers;
            }
            curl_setopt_array($ch, $options);

            $body   = curl_exec($ch);
            $status = (int)curl_getinfo($ch, CURLINFO_HTTP_CODE);
            $err    = curl_error($ch);
            curl_close($ch);

            if (!is_string($body) || $err !== '') {
                $lastError = "ulanish xatosi: {$err}";
                continue;                                   // tarmoq — qayta urinamiz
            }
            if ($status >= 500) {
                $lastError = "xizmat xatosi (HTTP {$status})";
                continue;                                   // server — qayta urinamiz
            }

            $data = json_decode($body, true);
            if (!is_array($data)) {
                $lastError = 'javob JSON emas';
                continue;
            }
            if ($status >= 400) {
                // Mijoz xatosi (noto'g'ri fayl va h.k.) — qayta urinish foydasiz
                return RecognitionResult::fromArray([
                    'valid' => false,
                    'error' => isset($data['detail']) ? (string)$data['detail'] : "HTTP {$status}",
                ]);
            }

            $result = RecognitionResult::fromArray($data);

            // Ilova xizmatnikidan qattiqroq chegara qo'yishi mumkin.
            if ($result->valid && $this->minConfidence > 0.0 && $result->confidence < $this->minConfidence) {
                return RecognitionResult::fromArray([
                    'valid'      => false,
                    'error'      => 'Low recognition confidence (client threshold)',
                    'confidence' => $result->confidence,
                    'characters' => $result->characters,
                ]);
            }

            return $result;
        }

        return RecognitionResult::fromArray(['valid' => false, 'error' => $lastError]);
    }
}
